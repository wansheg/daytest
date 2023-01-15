#include "afpga/LinkAllPasses.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/XILINXFunctionInfoUtils.h"
#include "llvm/Analysis/XILINXLoopInfoUtils.h"

#include "llvm/Transforms/Scalar.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/XILINXHLSIRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/XILINXLoopUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/XILINXLoopInfoUtils.h"
#include "afpga/Diagnostics/Diagnostic.h"
#include "afpga/Support/Metadata/FuncPragmaAttr.h"
#include "afpga/Support/Metadata/LoopPragmaAttr.h"

#include "afpga/afpgaConfig.h"

#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "auto-loop-pipeline"

#define UNROLL_CODE_SIZE_LIMIT afpgaConfig::GlobalConfig().MaxUnrollThreshold
#define PIPELINE_LIMIT   afpgaConfig::GlobalConfig().AutoPipelineLoopsThreshold

namespace {

struct LoopNode;
struct LoopNest{
  enum NestType{
    CallNest,
    PerfectNest,
    NatualNest
  };
  NestType nest;
  LoopNode *outer;
  LoopNode *inner;

  LoopNest(LoopNode *outer, LoopNode *inner, NestType nest ) :
    outer(outer),
    inner(inner),
    nest(nest)
  {  }
};
struct LoopNode{
  enum PipelineStatus { Unknown = 0, PipelineDisabled, PipelineEnabled, DataflowPipeline };


  PipelineStatus pipeline;

  enum AutoPipelineStatus{ Unset = 0, Prohibit = 1, AutoEnable = 2 };
  AutoPipelineStatus autoPipeline;

  int64_t unroll_factor;  // 1 means unroll off , -1ull means complete unroll,  0 means no specified  unroll pragma

  uint64_t trip_count;

  //most of time the parent size is 1,
  //but according function call , one loop can be nested by multi parent loop
  SmallVector<LoopNest*, 1>  inner;
  SmallVector<LoopNest*, 1>  outer;

  BasicBlock *loopheader;

  LoopNode( BasicBlock *loopHeader,
      PipelineStatus pipeline_status,
      uint64_t trip_count ,
      uint64_t unroll_factor):
    pipeline(pipeline_status),
    loopheader(loopHeader),
    trip_count(trip_count),
    autoPipeline(LoopNode::Unset),
    unroll_factor(unroll_factor)
  {
  }

  bool isTopLoop( ) {
    return !outer.size();
  }

  uint64_t getFlattenTripCount()
  {
    if ( trip_count == -1ull ) {
      return -1ull;
    }

    uint64_t sum = 0;
    for(LoopNest * nest: inner) {
      LoopNode * child = nest->inner;
      uint64_t child_flatten_tripcount = child->getFlattenTripCount();
      if (child_flatten_tripcount == -1ull)
        return -1ull;
      else
        sum += child_flatten_tripcount;
    }
    if (sum == 0) {
      return trip_count;
    }
    else
      return sum * trip_count;
  }
};


struct CallLoopNest : public LoopNest{
  SmallVector<CallInst*, 4> calls;

  CallLoopNest(LoopNode* outer, LoopNode *inner,  const SmallVectorImpl<CallInst*> &calls): LoopNest(outer, inner, LoopNest::CallNest) , calls(calls.begin(), calls.end()) {
  }
};


struct PerfectLoopNest: public LoopNest{
  PerfectLoopNest(LoopNode* outer, LoopNode *inner): LoopNest(outer, inner, LoopNest::PerfectNest)  { }
};

struct NatualLoopNest: public LoopNest{
  NatualLoopNest(LoopNode* outer, LoopNode *inner): LoopNest(outer, inner, LoopNest::NatualNest) {  }
};


typedef std::map<Function* , SmallVector<LoopNode*, 4> > Function2LoopsMap;
typedef std::map<Function*, SmallVector<CallInst*, 4> > Function2CallsMap;

class AutoPipeline : public ModulePass {
  CallGraph *CG;
  afpgaModuleDiagnostic *RD;

  DenseMap<Function*, ScalarEvolution*> func2SE;
  DenseMap<Function*, LoopInfo*> func2LoopInfo;
  DenseMap<Function*, DominatorTree*> func2DT;
  DenseMap<Function*, PostDominatorTree*> func2PDT;
  DenseMap<Function*, AssumptionCache*> func2AC;

  DenseMap<BasicBlock*, LoopNode*> mLoop2Info;

public:
  static char ID; // Pass ID, replacement for typeid

  AutoPipeline() : ModulePass(ID) {
    initializeAutoPipelinePass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
  LoopNode* recurVisitLoop( Loop *loop, ScalarEvolution* SE) ;
  void deletePipelineOffMetadata(Loop *loop );

  LoopNode * buildLoopNode(Loop *curLoop) ;
  void buildPerfectLoopNest(LoopNode *outer, LoopNode *inner) ;
  void buildNatualLoopNest(LoopNode* outer, LoopNode* inner);
  void buildCallLoopNest(LoopNode* outer, LoopNode* inner, const SmallVectorImpl<CallInst*> &call) ;
  void recurBuildCallLoopNest(LoopNode* curNode, SmallVector<CallInst*, 4> callNest, Function2LoopsMap func2Loops, Function2CallsMap func2Calls );

  void propagateInFullUnroll(LoopNode* node);
  void propagateInProhibit(LoopNode* node);
  void propagateOutProhibit(LoopNode* node);
  bool isStableLoopWrapped(LoopNode* node);
  void updateAutoPipeline(LoopNode* node);

  LoopInfo *getLoopInfo(Function *func);
  ScalarEvolution *getScalarEvolution(Function* func) ;
  DominatorTree* getDominatorTree(Function *func);
  PostDominatorTree* getPostDominatorTree( Function *func);



  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<CallGraphWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

};
} // namespace

char AutoPipeline::ID = 0;


/*
 * if the loop bound is predicatable ,
 *  return the tripcount ,
 *    if the tripcount is constant
 *      return constant value ,
 *    else
 *      return -1;
 * else if the loop tripcount is not predicatable ,
 *  return -1
 */
static uint64_t getLoopTripCount(const Loop *L, ScalarEvolution *SE){

#if 1

  BasicBlock *exitingBlock = L->getExitingBlock();

  if (!exitingBlock){
    return -1ull;
  }

  unsigned int tripcount = SE->getSmallConstantTripCount(L, exitingBlock);
  if (tripcount == 0 ) {
    return -1ull;
  }
  else {
    return tripcount;
  }
#else
  const SCEV *tripCount = SE->getMaxBackedgeTakenCount(L);
  DEBUG( dbgs() << "tripcount of <" << getLoopName(L).getValueOr("Unknown") << "> = " ;
      tripCount->dump();
      );


  if (tripCount && isa<SCEVConstant>(tripCount)) {
    tripCount = SE->getAddExpr(tripCount, SE->getOne(tripCount->getType()));
    return cast<SCEVConstant>(tripCount)->getAPInt().getZExtValue();
  }
  else {
    return -1ull;
  }
#endif
}


uint64_t getUnrollInfo( Loop *curLoop )
{
  if (isFullyUnroll(curLoop)) {
    return -1ull;
  }
  else if (isUnrollOff(curLoop)) {
    return 1;
  }
  else if (auto factor = getUnrollFactorUInt64(curLoop)) {
    return factor;
  }
  else {
    return 0;
  }
}

LoopNode* AutoPipeline::buildLoopNode(Loop *curLoop)
{

  if (mLoop2Info.count(curLoop->getHeader())) {
    return mLoop2Info[curLoop->getHeader()];
  }

  LoopNode::PipelineStatus pipeline = LoopNode::Unknown;
  if (isPipeline(curLoop)) {
    pipeline =  LoopNode::PipelineEnabled  ;
  }
  else if (isDataFlow(curLoop)) {
    pipeline = LoopNode::DataflowPipeline;
  }
  else if (isPipelineOff(curLoop))  {
    pipeline = LoopNode::PipelineDisabled;
  }
  uint64_t unroll_factor = getUnrollInfo(curLoop);

  ScalarEvolution *SE = getScalarEvolution(curLoop->getHeader()->getParent());

  uint64_t curTripcount = getLoopTripCount(curLoop, SE);


  LoopNode *node = new LoopNode(curLoop->getHeader(), pipeline, curTripcount, unroll_factor);

  mLoop2Info[curLoop->getHeader()] = node;

  return node;
}

void AutoPipeline::buildNatualLoopNest(LoopNode* outer, LoopNode* inner)
{

  NatualLoopNest * nest = new NatualLoopNest(outer, inner);
  outer->inner.push_back(nest);
  inner->outer.push_back(nest);
}
void AutoPipeline::buildCallLoopNest(LoopNode* outer, LoopNode* inner, const SmallVectorImpl<CallInst*> &calls)
{
  CallLoopNest *nest = new CallLoopNest(outer, inner, calls) ;
  outer->inner.push_back(nest);
  inner->outer.push_back(nest);
}

void AutoPipeline::buildPerfectLoopNest(LoopNode* outer, LoopNode *inner)
{
  PerfectLoopNest *nest = new PerfectLoopNest(outer, inner);
  outer->inner.push_back(nest);
  inner->outer.push_back(nest);
}


void AutoPipeline::deletePipelineOffMetadata(Loop *loop )
{
  //now, we have collect all loop pipeline  information, can delete the PipelineOff pragma
  //for perfect nest Loop, the innerMostLoop will come into here also , so it is assumed to delete
  //all pipelineOff  for all loops
  if (LoopPragmaPipeline::HasPipelineOff(loop)) {

    //delete Pipeline Off metadata to loop

    MDNode *LoopID = loop->getLoopID();
    assert(LoopID &&"unexpected");

    assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
    assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

    SmallVector<Metadata*, 4> Args;

    for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; ++i) {
      Metadata *metadata = LoopID->getOperand(i);
      MDNode *MD = dyn_cast<MDNode>(metadata);
      if (MD && isa<MDString>(MD->getOperand(0))) {
        MDString* tag = cast<MDString>(MD->getOperand(0));
        if (tag->getString().equals("llvm.loop.pipeline.enable")) {
          continue;
        }
      }
      Args.push_back(metadata);
    }

    LLVMContext &Ctx = loop->getHeader()->getParent()->getContext();
    LoopID = MDNode::get(Ctx, Args);

    LoopID->replaceOperandWith(0, LoopID);
    loop->setLoopID(LoopID);
  }
}


LoopNode* AutoPipeline::recurVisitLoop(Loop *loop, ScalarEvolution *SE)
{
  Loop * curLoop = loop;
  std::string info;

  LoopNode *node = buildLoopNode(loop);
  LoopNode *curNode = node;
  while (isPerfectNestLoop(curLoop, SE, info)) {
    //check perfect nesting Loop
    DEBUG( dbgs() << info << "\n");

    Loop * subLoop = *curLoop->begin();

    if (isPipeline(curLoop) || isPipelineOff(curLoop) || isDataFlow(curLoop) || hasUnrollEnableMetadata(curLoop)) {
      DEBUG(dbgs() << "current loop " << getLoopName(curLoop).getValueOr("unknown")
          << "has been set pipeline, so  doesn't consider perfectNest for subLoop "
          << getLoopName(subLoop).getValueOr("unknown") << "\n"; );
      break;
    }
    LoopNode *subLoopNode = buildLoopNode(subLoop);
    buildPerfectLoopNest(curNode, subLoopNode);

    curNode = subLoopNode;
    curLoop = subLoop;
  }

  DEBUG(
    if (!info.empty()) {
      dbgs() << info << "\n";
    }
  );



  for (Loop *subLoop : *curLoop) {
    LoopNode *subNode = recurVisitLoop(subLoop, SE);
    buildNatualLoopNest(curNode, subNode);
  }
  deletePipelineOffMetadata(curLoop);
  return node;
}


void AutoPipeline::recurBuildCallLoopNest(LoopNode* curNode, SmallVector<CallInst*, 4> callNest, Function2LoopsMap func2Loops, Function2CallsMap func2Calls )
{
  Function *callee = callNest.back()->getCalledFunction();

  for(LoopNode *subNode:  func2Loops[callee]) {
    buildCallLoopNest(curNode, subNode, callNest);
  }
  for( CallInst *call: func2Calls[callee]) {
    callNest.push_back(call);
    recurBuildCallLoopNest(curNode, callNest, func2Loops, func2Calls);
  }
}

void AutoPipeline::propagateInFullUnroll(LoopNode *node)
{
  //pipeline function can cause fully unroll  for inner loop in the function
  for( LoopNest* nest : node->inner) {
    nest->inner->unroll_factor = -1;
    propagateInFullUnroll(nest->inner);
  }
}

void AutoPipeline::propagateInProhibit(LoopNode *node)
{
  for( LoopNest* nest : node->inner) {
    nest->inner->autoPipeline = LoopNode::Prohibit;
    propagateInProhibit(nest->inner);
  }
}
void AutoPipeline::propagateOutProhibit(LoopNode* node)
{
  for( LoopNest *nest: node->outer) {
    nest->outer->autoPipeline = LoopNode::Prohibit;
    propagateOutProhibit(nest->outer);
  }
}

//if the loop is nested in "dataflowLoop" or 'autoPipeline == Prohibit'
//we take the loop  nested by 'stable loop' ,
//stable loop means the loop can be  'dataflow' or
//'won't be complete unroll implicitly by pipeline '
bool AutoPipeline::isStableLoopWrapped(LoopNode *node)
{
  for( LoopNest *nest: node->outer) {
    if (nest->outer->autoPipeline == LoopNode::Prohibit ||
        nest->outer->pipeline == LoopNode::DataflowPipeline) {
      return true;
    }
  }
  return false;
}

void AutoPipeline::updateAutoPipeline(LoopNode *node)
{
  //skip perfect nest
  while (node->inner.size() == 1 && node->inner[0]->nest == LoopNest::PerfectNest) {
    LoopNode *inner = node->inner[0]->inner;
    node = inner;
  }
  propagateOutProhibit(node);
  node->autoPipeline = LoopNode::AutoEnable;

  LoopInfo *LI = getLoopInfo(node->loopheader->getParent());
  Loop *loop = LI->getLoopFor(node->loopheader);

  MDNode *LoopID = loop->getLoopID();
  if (!LoopID)
    return ;

  RD->emitAutoPipelineInfo(loop);

  assert(LoopID->getNumOperands() > 0 && "requires at least one operand");
  assert(LoopID->getOperand(0) == LoopID && "invalid loop id");

  SmallVector<Metadata*, 4> Args( LoopID->operands());


  LLVMContext &Ctx = node->loopheader->getContext();
  Metadata *Vals[] = {
      MDString::get(Ctx, "llvm.loop.pipeline.enable"),
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), -1)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt1Ty(Ctx), 0)),
      ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt8Ty(Ctx), -1)),
      MDString::get(Ctx, "auto")
  };

  Args.push_back(MDNode::get(Ctx, Vals));
  LoopID = MDNode::get(Ctx, Args);

  LoopID->replaceOperandWith(0, LoopID);
  loop->setLoopID(LoopID);

}

ScalarEvolution* AutoPipeline::getScalarEvolution(Function *F)
{
  if (func2SE.end() != func2SE.find(F))  {
    return func2SE[F];
  }

  auto *DT = getDominatorTree(F);
  auto *PDT = getPostDominatorTree(F);
  auto *LI = getLoopInfo(F);


  AssumptionCache *AC = nullptr;
  if (func2AC.count(F)) {
    AC = func2AC[F];
  }
  else {
    AC = new AssumptionCache(*F);
    func2AC[F] = AC;
  }

  TargetLibraryInfo &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();

  auto SE = new ScalarEvolution(*F, TLI, *AC, *DT, *LI);
  func2SE[F] = SE;
  return SE;
}

LoopInfo * AutoPipeline::getLoopInfo(Function *F)
{
  if (func2LoopInfo.count(F) ) {
    return func2LoopInfo[F];
  }
  auto DT = getDominatorTree(F);
  auto LI = new LoopInfo(*DT);
  func2LoopInfo[F] = LI;
  return LI;
}

DominatorTree * AutoPipeline::getDominatorTree(Function *F)
{
  if (func2DT.count(F)) {
    return func2DT[F];
  }

  auto DT = new DominatorTree(*F);
  auto PDT = new PostDominatorTree;
  PDT->recalculate(*F);

  func2DT[F] = DT;
  func2PDT[F] = PDT;
  return DT;
}

PostDominatorTree * AutoPipeline::getPostDominatorTree(Function *F)
{
  auto* DT = getDominatorTree(F);
  (void)(DT);
  assert(func2PDT.count(F) && "unexpected");
  return func2PDT[F];
}

bool AutoPipeline::runOnModule(Module &M)
{
  if (PIPELINE_LIMIT <= 0) {
    return false;
  }

  afpgaModuleDiagnostic diag(DEBUG_TYPE, M);
  RD = &diag;

  CallGraph &CGRef = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  CG = &CGRef;
  ReversePostOrderTraversal<CallGraphNode*>  callOrder(CGRef.getExternalCallingNode());
  SmallVector<CallGraphNode*, 4> calls( callOrder.begin(), callOrder.end());
  SmallVector<CallGraphNode*, 4> reverseOrder;
  for( int i = calls.size() -1; i >= 0; i--) {
    if (!calls[i]->getFunction()) {
      continue;
    }
    Function * func = calls[i]->getFunction();

    if(!func || func->isDeclaration()) {
      continue;
    }

    if (func->hasFnAttribute("fpga.wrapper.func")) {
      continue;
    }

    if (HasVivadoIP(func)) {
      continue;
    }

    reverseOrder.push_back(calls[i]);
  }


  //if the callInst's callee function doesn't contain loop, it should not be insert
  //into 'callInFunctionNotLoopNest'
  Function2LoopsMap loopInFunction;
  Function2CallsMap callInFunctionNotLoopNest;
  SmallSetVector<Function*, 4> funcWithDataflow;
  //from bottom to top
  for (auto *callNode : reverseOrder) {
    Function * func = callNode->getFunction();

    LoopInfo *LI = getLoopInfo(func);
    ScalarEvolution* SE = getScalarEvolution(func);
    SmallVector<LoopNode*, 4> nodes;
    for( Loop * loop : *LI) {
      LoopNode * node = recurVisitLoop(loop, SE);
      loopInFunction[func].push_back(node);
    }
    //check subcall that contain loop
    for( auto &callRecord : *callNode){
      assert(isa<CallInst>((Value*)callRecord.first) && "unexpected" );
      CallInst * callInst = cast<CallInst>((Value*)callRecord.first) ;
      Function* callee = callRecord.second->getFunction();
      Loop* parentLoop = LI->getLoopFor(callInst->getParent());
      if (parentLoop) {
        assert(mLoop2Info.count(parentLoop->getHeader()) && "unexpected");
        LoopNode *parentNode = mLoop2Info[parentLoop->getHeader()];
        //process 'loopInFunction[callee]'
        //
        for(auto *calleeLoop : loopInFunction[callee]) {
          buildCallLoopNest(parentNode, calleeLoop, SmallVector<CallInst*, 1>({ callInst }) );
        }

        for(auto* callInst : callInFunctionNotLoopNest[callee]) {
          SmallVector<CallInst*, 4> calls({callInst});
          recurBuildCallLoopNest(parentNode, calls, loopInFunction, callInFunctionNotLoopNest);
        }
        if (funcWithDataflow.count(callee)){
          parentNode->autoPipeline = LoopNode::Prohibit;
          propagateOutProhibit(parentNode);
        }
      }
      else {
        //it is a call not in any loop in the current function ,
        if (callInFunctionNotLoopNest.count(callee) ||
            loopInFunction.count(callee)) {
          callInFunctionNotLoopNest[func].push_back(callInst);
        }
      }

      if (funcWithDataflow.count(callee)) {
        funcWithDataflow.insert(func);
      }
    }

    if (isDataFlow(func)) {
      funcWithDataflow.insert(func);
    }
    else if (isPipeline(func)) {
      SmallVector<Function*, 4> worklist({func});
      while( !worklist.empty()) {
        Function *xf = worklist.back();
        worklist.pop_back();
        for( auto *node: loopInFunction[xf]) {
          node->unroll_factor = -1;
          propagateInFullUnroll(node);
          //propagateOutProhibit(node);
        }
        for(auto *call : callInFunctionNotLoopNest[xf]) {
          Function *callee = call->getCalledFunction();
          worklist.push_back(callee);
        }
      }
    }
  }
  //finish build LoopNestGraph,
  //and , for all inner loop in pipeline function , the inner loop
  //is implicit full-unroll ;
  //for all outer loop of 'dataflow function', the outer loop of
  //'dataflow function' is autoPipeline prohibit ;

  //get all Top , leaf Loop
  SmallVector<LoopNode*, 4> leafLoops;
  SmallVector<LoopNode*, 4> topLoops;
  for ( auto kv : mLoop2Info) {
    LoopNode *node = kv.second;
    if (!node->outer.size())
      topLoops.push_back(node);
    if (!node->inner.size()) {
      leafLoops.push_back(node);
    }
  }

  //check all LoopNode,
  //1. if 'pipelineStatus == PipelineEnabled', set all inner and outer loop's
  //   'autoPipelineStatus" as   "Prohibit'
  //2. if 'pipelineStatus == PipelineDisabled' or "pipelineStatus == DataflowPipeline"
  //   set all outer loop's 'autoPipelineStatus' as 'Prohibit'
  //3. if 'unrollFactor > 0 ', set all outer loop's 'autoPipelineStatus' as 'prohibit'
  //
  //this will prevent all outer loop from pipeline;
  for ( auto kv : mLoop2Info) {
    //BasicBlock *head = kv.first;
    LoopNode *node = kv.second;
    if (node->pipeline == LoopNode::PipelineEnabled) {
      propagateInProhibit(node);
      propagateOutProhibit(node);
      node->autoPipeline = LoopNode::Prohibit;
    }
    if (node->pipeline == LoopNode::PipelineDisabled ||
        node->pipeline == LoopNode::DataflowPipeline) {
      propagateOutProhibit(node);
      node->autoPipeline = LoopNode::Prohibit;
    }
    if (node->unroll_factor > 0) {
      propagateOutProhibit(node);
    }
  }

  SmallVector<LoopNode*, 4> worklist(leafLoops);
  SmallSet<LoopNode*, 32> visited;

  while (!worklist.empty()){
    LoopNode *node = worklist.back();
    worklist.pop_back();
    visited.insert(node);

    if (node->autoPipeline == LoopNode::Prohibit) {
      continue;
    }

    if (node->unroll_factor == -1) {
      //for fully unroll, visit outer loop

    }
    else if (node->getFlattenTripCount() > PIPELINE_LIMIT) {
      updateAutoPipeline(node);
      propagateOutProhibit(node);
      continue;
    }
    else if (node->isTopLoop()){
      updateAutoPipeline(node);
      propagateOutProhibit(node);
      continue;
    }
    else if (isStableLoopWrapped(node)) {
      updateAutoPipeline(node);
      propagateOutProhibit(node);
      for (LoopNest* parentEdge: node->outer) {
        LoopNode * outer = parentEdge->outer;

        for (LoopNest *childEdge: outer->inner) {
          LoopNode *inner = childEdge->inner;
          if (visited.count(inner) ){
            if (inner->autoPipeline == LoopNode::Unset) {
              inner->autoPipeline = LoopNode::AutoEnable;
            }
          }
        }//for
      }//for
      continue;
    }
    else if (node->unroll_factor > 0 ) {
      updateAutoPipeline(node);
      propagateOutProhibit(node);
      continue;
    }
    //if get outer loops of current loop , and check
    //1. if all  inner loop of  the outer loop
    //is visited , then,  push the outer loop into 'worklist',
    //2. otherwise do nothing

    for (LoopNest* parentEdge: node->outer) {
      LoopNode * outer = parentEdge->outer;

      bool allInnerLoopsVisited = true;
      for (LoopNest *childEdge: outer->inner) {
        LoopNode *inner = childEdge->inner;
        if (!visited.count(inner) ){
          allInnerLoopsVisited = false;
          break;
        }
      }
      if (allInnerLoopsVisited) {
        worklist.push_back(outer);
      }
    }
  }
  return false;
}



INITIALIZE_PASS_BEGIN(AutoPipeline, DEBUG_TYPE,
                      "afpga pipeline loop automatically",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)

INITIALIZE_PASS_END(AutoPipeline, DEBUG_TYPE,
                      "afpga pipeline loop automatically",
                    false,
                    false)

Pass *llvm::createAutoPipelinePass() {
  return new AutoPipeline();
}

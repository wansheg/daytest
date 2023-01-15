#include "afpga/LinkAllPasses.h"
#include "afpga/Support/Vectorization.h"
#include "afpga/AFPGAConfig.h"
#include "afpga/Diagnostics/Diagnostic.h"
#include "afpga/TransformUtils/MiscUtil.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "afpga/Support/Mangle.h"
#include "llvm/Analysis/XILINXLoopInfoUtils.h"
#include "llvm/Analysis/XILINXFunctionInfoUtils.h"

#include <map>
#include <algorithm>
#include <utility>
#include <unordered_map>

//#define NEW_AUTOINLINE_FEATURE_20_1
// We decide to disable ESTIMATEION_BASED_PIPELINE, because pipeline unrolling happens in afpga now
//#define ESTIMATEION_BASED_PIPELINE


using namespace llvm;

#define DEBUG_TYPE "afpga-inliner"

namespace {

  ///////////////////////////////////////////
  //   callSite info struct //
  ///////////////////////////////////////////

typedef enum {
    STOP_INLINE         = 0x0,      // Not do auto inline
    MUST_INLINE         = 0x1,      // Final decision: must do inline without any condition. Here most belongs to 19.2 feature: small callee functions.
    NEED_CHECK_INLINE   = 0x2       // Maybe do inline, need another condition to do final double-check
} InlineDecision;


typedef struct CallSiteInformation {
  CallSite CS;
  int callee_size;
  int caller_size;
  int call_num;
  InlineDecision decision;
} CallSiteInfo;

// Use callee function size for sort order
bool LessCallSiteInfo(const CallSiteInfo& s1, const CallSiteInfo& s2)
{
  return s1.callee_size < s2.callee_size;
}


  ///////////////////////////////////////////
  //   auto inline main class              //
  ///////////////////////////////////////////

/// \brief Inliner pass based on cost model in AFPGA
class AFPGAInliner : public llvm::ModulePass {

public:
  AFPGAInliner() : ModulePass(ID) {
    initializeAFPGAInlinerPass(*PassRegistry::getPassRegistry());
  }
  static char ID; // Pass identification, replacement for typeid

  std::set<Function *> InlinedCallees;
  std::map<Function *, int> FuncSizeMap; //record function size for each callee

  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const {
    AU.addRequired<CallGraphWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
  }

  virtual bool runOnModule(llvm::Module &M);
  virtual bool runOnFunction(llvm::Function &F, AFPGAModuleDiagnostic &RD);
  bool clearDeadFunctions(llvm::Module &M);

  void getValidCallSites(Function& F, std::vector<CallSiteInfo>& callsites);
  bool ActionInline(CallSite CS, AFPGAModuleDiagnostic &RD,
                    std::unordered_multimap<std::string, std::string> &IMap);
  // true: always inline; false: noinline
  bool getInlineCost(CallSiteInfo &CSInfo);

  ///////////////////////////////////////////
  //   Inline conditions below             //
  ///////////////////////////////////////////

  // check conflicts with other pragmas
  bool checkPragmaConflictInline(CallSite CS, std::string &ErrMsg);
  bool checkFunctionConflictInline(Function &F, std::string &ErrMsg);
  bool checkDataflowConflictInline(CallSite CS, std::string &ErrMsg);
  bool checkRewindPipelineConflictInline(Function *callee, std::string &ErrMsg);
  bool loopDataflowExist(Loop &L);

  // auto inline condition
  int getNumCallFunc(Function *callee, Function *caller, CallSite CS);
  int getFuncSize(Function *F);
  int increaseCodeSize(CallSiteInfo CSInfo);
  bool satisfyAutoInline(CallSiteInfo &CSInfo);
  bool specialCalleeMustInline(CallSite CS);

  // pipeline related
  void collectPipelineLoops(Loop *L, std::map<Loop*, int> &loops, bool pipeline_flag, int count, DominatorTree &DT, LoopInfo &LI);
  void PrepareInfoFromPipeline(Function *F, std::map<BasicBlock*, int> &BBMapCount);
  int  upwardPipelineCount(Function *F, BasicBlock* start);

  // judge loop pragmas
  bool LoopMayDisappear(Loop *L, DominatorTree &DT, LoopInfo &LI);
  int  getLoopTripCount(Loop *L, DominatorTree &DT, LoopInfo &LI);

  // auto-rom-infer related
  // only for guard variable, we should keep the IR pattern, so that following
  // auto-rom-infer could optimize the 'init.check' BB
  bool isSpecailPatternAutoRomInitCheck(BasicBlock* InitBB, BranchInst *BI);
  BasicBlock *getInitialBB(BranchInst *BI, GlobalVariable *&LocalStatic, std::string &GuardName);
  bool shouldReserveForStaticInit(BasicBlock *BB);
  bool shouldReserveForStaticInit(Instruction *I);
  bool shouldReserveForStaticInit(CallSite CS);
  bool isStaticGuardVarInitCheck(CallSite CS);

private:
  // Threshold default value is 100.  It can be obtained via '-auto-inline-callee-size=100'
  const int Threshold_deta_callee_size = AFPGAConfig::GlobalConfig().AutoInlineCalleeThreshold;
  // Threshold default value is 4000. It can be obtained via '-auto-inline-caller-size=4000'
  const int Threshold_max_caller_size  = AFPGAConfig::GlobalConfig().AutoInlineCallerThreshold;

};
} // namespace

char AFPGAInliner::ID = 0;
INITIALIZE_PASS_BEGIN(AFPGAInliner, DEBUG_TYPE, "Automatic inliner in afpga",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(AFPGAInliner, DEBUG_TYPE, "Automatic inliner in afpga",
                    false, false)


Pass *llvm::createAFPGAInlinerPass() { return new AFPGAInliner(); }


bool AFPGAInliner::runOnModule(Module& M) {
  bool Changed = false;
  AFPGAModuleDiagnostic RD(DEBUG_TYPE, M);

  // 1. Walk through the SCC, bottom->up.
  std::vector<Function*> funcs;
  CallGraph& call_graph = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  for (scc_iterator<CallGraph *> si = scc_begin(&call_graph); !si.isAtEnd(); ++si) {
      // Visit all functions in the current SCC.
      const std::vector<CallGraphNode *> &SCC = *si;
      for (unsigned i = 0; i < SCC.size(); ++i)
          if (Function* F = SCC[i]->getFunction()) {
              funcs.push_back(F); // record function list in scc order
              FuncSizeMap[F] = getFuncSize(F); //record its size
          }
  }

  // 2. for each function, auto inline
  for (unsigned i = 0; i < funcs.size(); i++)
      if (!funcs[i]->isDeclaration())
          Changed |= runOnFunction(*funcs[i], RD);

  // 3. clear dead functions
  Changed |= clearDeadFunctions(M);
  return Changed;
}


bool AFPGAInliner::clearDeadFunctions(Module& M) {
  bool Changed = false;

  for (Function *F : InlinedCallees) {
    Function *Callee = F;
    if (Callee && Callee->use_empty() && Callee->hasLocalLinkage()) {
      Callee->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}


bool AFPGAInliner::runOnFunction(Function& F, AFPGAModuleDiagnostic &RD) {
  bool Changed = false;
  // do nothing for wrapper function
  if (F.hasFnAttribute("fpga.wrapper.func"))
    return false;

  DEBUG(errs() << "Inliner visiting Function:"
        << F.getName() << "\n");

  // Step 1: collect callsite list which could do 'auto inline'
  std::vector<CallSiteInfo> CallSites;
  getValidCallSites(F, CallSites);

  // Step 2: action inline for each selected callsite
  int caller_size = 0;
  bool need_update_caller_size = true;

  // Record the successful inlining pairs <callee info, caller name>.
  std::unordered_multimap<std::string, std::string> InlinerMap;

  for (unsigned idx = 0; idx != CallSites.size(); ++idx) {
    CallSite CS = CallSites[idx].CS;
    if (need_update_caller_size)
      caller_size = FuncSizeMap[&F];
#ifdef NEW_AUTOINLINE_FEATURE_20_1
    // no need to do inline
    if (CallSites[idx].decision == STOP_INLINE)
      continue;
    // Just go ahead to action inline
    if (CallSites[idx].decision == MUST_INLINE) {
      DEBUG(errs() << "19.2 Feature: small callee function. "
             << "Action inline without any fututher checking."
             << " Call: " << *CS.getInstruction() << "\n");
    }
    // If exceed 'max threshold', stop auto inline for current CS
    if (CallSites[idx].decision == NEED_CHECK_INLINE)
      if (caller_size < 0
       || CallSites[idx].callee_size < 0
       || caller_size + CallSites[idx].callee_size > Threshold_max_caller_size)
        continue;
#endif
    // Action Inline really
    DEBUG(errs() << "    Inlining: inline directive"
        << ", Call: " << *CS.getInstruction());
    // Attempt to inline the function...
    need_update_caller_size = ActionInline(CS, RD, InlinerMap);
    Changed |= need_update_caller_size;
    if (need_update_caller_size) {
      // (a) update caller size since INLINE actions
      int new_caller_size = getFuncSize(&F);
      FuncSizeMap[&F] = new_caller_size;
      // (b) update caller_size value for following CallSite, for consistance
      for (unsigned idx2 = idx + 1; idx2 < CallSites.size(); ++idx2)
        CallSites[idx2].caller_size = new_caller_size;
    }
  }

  // step 3: clear & reset before step into next function
  CallSites.clear();
  return Changed;
}


// collect callsites which could do auto inline
void AFPGAInliner::getValidCallSites(Function& F, std::vector<CallSiteInfo>& callsites) {
  std::vector<CallSiteInfo> original_cs;

  //1. collect original callsite list
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (CallSite CS = CallSite(&I)) {
        if (CS.getInstruction() &&
            CS.getCalledFunction() &&
            !CS.getCalledFunction()->isDeclaration()) {
          Function *callee = CS.getCalledFunction();
          Function *caller = &F;
          // caculate required info:
          CallSiteInfo cs_info;
          cs_info.CS          = CS;
          cs_info.callee_size = FuncSizeMap[callee];
          cs_info.caller_size = FuncSizeMap[caller];
          cs_info.call_num    = getNumCallFunc(callee, caller, CS);
          cs_info.decision    = STOP_INLINE; //default: not do inline for CS
          original_cs.push_back(cs_info);
        }
      }
    }
  }

  //2.filter: basic auto inline conditions
  for (unsigned i = 0; i < original_cs.size(); ++i) {
    if (getInlineCost(original_cs[i]))
      callsites.push_back(original_cs[i]);
  }
  original_cs.clear();

#ifdef NEW_AUTOINLINE_FEATURE_20_1
  //3.sort order: small->big callee func size
  std::sort(callsites.begin(), callsites.end(), LessCallSiteInfo);
#endif

  return;
}

// do Inline for each callsite
bool AFPGAInliner::ActionInline(CallSite CS, AFPGAModuleDiagnostic &RD,
    std::unordered_multimap<std::string, std::string> &IMap) {
  InlineFunctionInfo info;
  DebugLoc DLoc = CS->getDebugLoc();
  BasicBlock *Block = CS.getParent();
  Function *caller = Block->getParent();
  Function *callee = CS.getCalledFunction();

  // call native inline API: enable lifetime
  if (InlineFunction(CS, info, nullptr, true)) {
    InlinedCallees.insert(callee);
    if (afpga::canEmitMessage(callee, caller, IMap)) {
      AFPGADiagnostic &ORE= RD.getAFPGAFunctionDiagnostic(caller);
      ORE.emitAutoInline(caller, callee);
    }
    return true;
  }
  return false;
}





/// \brief Get the inline cost for the afpga-inliner.
///
bool AFPGAInliner::getInlineCost(CallSiteInfo &CSInfo) {
  bool const ALWAYS_INLINE = true;
  bool const NON_INLINE    = false;

  CallSite CS = CSInfo.CS;
  Function *Callee = CS.getCalledFunction();
  if (!Callee)
    return NON_INLINE;

  // Cannot inline if there is no function body
  if (Callee->isDeclaration())
    return NON_INLINE;

  if (!isInlineViable(*Callee))
    return NON_INLINE;

  // Do not inline if the users ask to not inline
  // Its priority is high, so that no need to check pragma conflict
  if (CS.hasFnAttr(Attribute::NoInline))
    return NON_INLINE;

  // Do inline if the user has inline pragma
  if (CS.hasFnAttr(Attribute::AlwaysInline)) {
    CSInfo.decision = MUST_INLINE;
    return ALWAYS_INLINE;
  }

  // Do inline if the user has No inline pragma, but callee is special
  if (specialCalleeMustInline(CS)) {
    CSInfo.decision = MUST_INLINE;
    return ALWAYS_INLINE;
  }

  // Conflict with other pragmas
  std::string ErrMsg = "Error: ";
  bool conflict = checkPragmaConflictInline(CS, ErrMsg);
  if (conflict) {
    //If inline pragma, error out directly
    if(CS.hasFnAttr(Attribute::AlwaysInline)) {
      //error out, Fixme
      llvm::errs()<<ErrMsg<<"\n";
    }
    //If auto inline, stop
    return NON_INLINE;
  }

  if (satisfyAutoInline(CSInfo))
    return ALWAYS_INLINE;

  return NON_INLINE;
}




// This function is used to judge special callee, which has no inline pragma,
// but they must do inline
bool AFPGAInliner::specialCalleeMustInline(CallSite CS) {
  Function* callee = CS.getCalledFunction();
  if (!callee || callee->isDeclaration())
    return false;

  //Always inline if this function returns a pointer.
  if (isa<PointerType>(callee->getReturnType()))
    return true;

  // Special function: operator=, operator[]
  std::string callee_name = getDemangleName(callee->getName());
  if ((callee_name.find("::operator=")  != std::string::npos)
   || (callee_name.find("::operator[]") != std::string::npos)
     )
    return true;

  // Special function: constructor, destructor
  if (isConDestructor(callee).first)
    return true;

  // Else, false
  return false;
}

bool AFPGAInliner::checkPragmaConflictInline(CallSite CS, std::string &ErrMsg) {
  bool conflict = false;
  Function* callee = CS.getCalledFunction();
  if (callee && checkFunctionConflictInline(*callee, ErrMsg)) {
    conflict = true;
    return conflict;
  }

  if (callee && checkRewindPipelineConflictInline(callee, ErrMsg)) {
    conflict = true;
    return conflict;
  }

  conflict = checkDataflowConflictInline(CS, ErrMsg);
  return conflict;
}



bool AFPGAInliner::checkFunctionConflictInline(Function &F, std::string &ErrMsg) {
  bool conflict = false;
  if (F.isDeclaration())
    return conflict;
  if (F.hasFnAttribute("fpga.dataflow.func")            // function dataflow
   || F.hasFnAttribute("fpga.static.pipeline")          // function pipeline
   || F.hasFnAttribute("fpga.top.func")                 // top pragma
   || F.hasFnAttribute("fpga.latency")                  // latency (function)
   || F.hasFnAttribute("fpga.protocol")                 // protocol pragma
   || F.hasFnAttribute("fpga.exprbalance.func")         // expression_balance
   || F.hasFnAttribute("fpga.mergeloop")                // loop_merge on function
   || hasFunctionInstantiate(&F)                        // function_instantiate pragma
   || F.hasFnAttribute("fpga.occurrence")               // occurrence pragma
   || F.hasFnAttribute("fpga.region.inline")            // region inline
   || F.hasFnAttribute("fpga.region.inline.off")        // region inline off
   || F.hasFnAttribute("fpga.recursive.inline")         // recursive inline
   ) {
    conflict = true;
    ErrMsg += "Pragma conflict with Function attribute. \n";
    return conflict;
  }

  for (Use& use: F.uses()) {
    User* user = use.getUser();
    CallSite callsite(user);
    if (callsite && callsite.isBundleOperand(&use)) {
      //if function is used in "directive.scope.entry["xlx_function_allocation"] or "sideeffect["xlx_function_allocation"]").
      ErrMsg += "Pragma conflict with Allocation pragma ";
      return true;
    }
  }
  return conflict;
}

// Loop rewind pipeline has strict condition: only 1 top-level loop in this function.
// If not, rewind pipeline will not happen. Pls see LoopRewind.cpp.
// So if callee function has such rewind pipeline, we should not do inline for safety.
bool AFPGAInliner::checkRewindPipelineConflictInline(Function *callee, std::string &ErrMsg) {
  bool conflict = false;
  if (callee->isDeclaration())
    return conflict;

  auto &LI_callee = getAnalysis<LoopInfoWrapperPass>(*callee).getLoopInfo();
  for (auto *L : LI_callee) {
    auto *LoopID = L->getLoopID();
    if (!LoopID) continue;
    auto *PipelineMD = GetUnrollMetadata(LoopID, "llvm.loop.pipeline.enable");
    if (!PipelineMD) continue;
    auto Rewind = mdconst::extract<ConstantInt>(PipelineMD->getOperand(2))->getZExtValue();
    if (Rewind) {
      conflict = true;
      ErrMsg += "Pragma conflict with loop rewind pipeline. \n";
      return conflict;
    }
  }
  return conflict;
}


bool AFPGAInliner::checkDataflowConflictInline(CallSite CS, std::string &ErrMsg) {
  Instruction* callInst = CS.getInstruction();
  CallInst* call = dyn_cast<CallInst>(callInst);
  if (!call)
    return false;

  bool conflict = false;
  Function* callee = CS.getCalledFunction();
  Function* caller = callInst->getParent()->getParent();

  //1. check callee function contains dataflow or not
  if (callee->hasFnAttribute("fpga.dataflow.func")) {
    conflict = true;
    ErrMsg += "Pragma conflict with Function dataflow in callee function. \n";
    return conflict;
  }
  DominatorTree DT(*callee);
  LoopInfo LI(DT);
  for (auto *L : LI) {
    if (loopDataflowExist(*L)) {
      conflict = true;
      ErrMsg += "Pragma conflict with loop dataflow, which is defined in callee function. \n";
      return conflict;
    }
  }

  //2. check caller function contains dataflow or not
  DT.recalculate(*caller);
  LI.releaseMemory();
  LI.analyze(DT);
  if (caller->hasFnAttribute("fpga.dataflow.func")) {
    // If the callsite not in a loop, conflict will happen
    Loop *L = LI.getLoopFor(call->getParent());
    if (L == nullptr || LoopMayDisappear(L, DT, LI)) {
      conflict = true;
      ErrMsg += "Pragma conflict with Function dataflow in caller function. \n";
      return conflict;
    }
  }
  if (Loop *L = LI.getLoopFor(call->getParent())) {
    MDNode *LoopID = L->getLoopID();
    if (LoopID && GetUnrollMetadata(LoopID, "llvm.loop.dataflow.enable")) {
      conflict = true;
      ErrMsg += "Pragma conflict with loop dataflow, which is defined in caller function. \n";
      return conflict;
    }
  }

  return conflict;
}

static int LoopHintPragmaValue(MDNode *MD) {
  if (MD) {
    assert(
        MD->getNumOperands() >= 2 &&
        "loop unroll/pipeline/flatten hint metadata should have 2 operands.");
    int Count =
        mdconst::extract<ConstantInt>(MD->getOperand(1))->getSExtValue();
    return Count;
  }
  return 0;
}


// get loop tripcount
// If <0, means that cannot obtain valid tripcount info, e.g. multi-exit loops
int AFPGAInliner::getLoopTripCount(Loop *L, DominatorTree &DT, LoopInfo &LI) {
  if (!L) return -1;
  Function *F = L->getHeader()->getParent();
  auto SE = ScalarEvolution(
      *F, getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(),
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F), DT, LI);
  unsigned TripCount = SE.getSmallConstantTripCount(L);
  if (TripCount == 0) {
    // cannot obtain valid tripcount info, e.g. multi-exit loops
    return -1;
  }
  else {
    // SE.getSmallConstantTripCount' will caculate as do-while, so real tripcount will minus 1
    TripCount = TripCount - 1;
  }
  return TripCount;
}

bool AFPGAInliner::LoopMayDisappear(Loop *L, DominatorTree &DT, LoopInfo &LI) {
  if (!L) return true;
  int TripCount = getLoopTripCount(L, DT, LI);

  if (TripCount < 0) {
    // cannot obtain valid tripcount info, e.g. multi-exit loops
    // so here we will not care the tripcount impact.
    return false;
  }
  // 0. auto unroll when tripcount==1
  else if (TripCount == 0 || TripCount == 1) {
    // auto unroll will delete this loop automatically
    return true;
  }

  auto *LoopID = L->getLoopID();
  if (!LoopID) return false;

  // 1. complete unroll
  if (GetUnrollMetadata(LoopID, "llvm.loop.unroll.full"))
    return true;

  // 2. factor unroll
  if (GetUnrollMetadata(LoopID, "llvm.loop.unroll.count")) {
    int count = LoopHintPragmaValue(GetUnrollMetadata(LoopID, "llvm.loop.unroll.count"));
    DEBUG(errs() << "Unroll factor happens. LoopTripCount: "<<TripCount<<"   factor: "<<count<<"\n");
    if ((int)TripCount == count) {
      // although factor unroll, if factor == tripcount,
      // it also means 'complete unroll'
      return true;
    }
  }

  // 3. otherwise, it means normal loop:
  return false;
}


bool AFPGAInliner::loopDataflowExist(Loop &L) {
  auto *LoopID = L.getLoopID();
  if (!LoopID)
    return false;
  if (GetUnrollMetadata(LoopID, "llvm.loop.dataflow.enable")) {
    return true;
  }
  //sub loops
  for (auto *Child : L)
    loopDataflowExist(*Child);
  return false;
}


// loops: map used to record how many counts its body will be copied
// pipeline_flag: parent loop found pipeline pragma or not
// count: tell sub loop how many count in current loop level
void AFPGAInliner::collectPipelineLoops(Loop *L,
       std::map<Loop*, int> &loops, bool pipeline_flag, int count,
       DominatorTree &DT, LoopInfo &LI) {
  if (!L) return;
  // parent loop should found pipeline flag
  if (pipeline_flag) {
    pipeline_flag = true;
    // if parent loop count >0, go ahead
    // If parent loop <=0, it means pipeline exist and tripcount cannot caculate, then withdraw
    loops[L] = (count > 0) ? (count * getLoopTripCount(L, DT, LI)) : (count);
    count = loops[L];
  }
  // parent loop no help, continue to see current loop has pipeline or not
  else if (isPipeline(L)) {
    pipeline_flag = true;
    // pipeline unrolling only happen in sub-loops, so its count is 1
    loops[L] = 1;
    count = 1;
  }
  else {
    pipeline_flag = false;
    count = 1;
  }
  // sub loops
  std::vector<Loop*> SubLoops(L->begin(), L->end());
  for (unsigned i = 0, e = SubLoops.size(); i != e; ++i)
    collectPipelineLoops(SubLoops[i], loops, pipeline_flag, count, DT, LI);
  return;
}

void AFPGAInliner::PrepareInfoFromPipeline(Function *F,
                 std::map<BasicBlock*, int> &BBMapCount) {
  // loop map to count
  std::map<Loop*, int> LoopMapCount;
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  for (auto *L : LI)
    collectPipelineLoops(L, LoopMapCount, false, 1, DT, LI);

  // BB map to count
  for (Function::iterator bb = F->begin(); bb != F->end(); bb++) {
    BasicBlock *BB = &(*bb);
    Loop *L = LI.getLoopFor(BB);
    if (L && LoopMapCount.count(L) > 0)
      BBMapCount[BB] = LoopMapCount[L];
  }
  return;
}

// start from BB, exploring upward, to see if it is in a loop under pipeline region
// if yes, return count number after unrolling
// If found pipeline but cannot caculate tripcount, return -1
int AFPGAInliner::upwardPipelineCount(Function *F, BasicBlock* start) {
  if (!start || !F)
    return 0;
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  Loop *L = LI.getLoopFor(start);
  if (!L)
    return 0;

  std::vector<Loop *> loops;
  bool pipeline_flag = false;
  while (L) {
    if (isPipeline(L)) {
      pipeline_flag = true;
      break;
    }
    loops.push_back(L);
    L = L->getParentLoop();
  }
  if (!pipeline_flag)
    return 0;

  //After found pipeline, start to caculate count
  int count = 1;
  for (unsigned i = 0; i < loops.size(); ++i) {
    int tripcount = getLoopTripCount(loops[i], DT, LI);
    if (tripcount < 0)
      return -1;
    count = count * tripcount;
  }
  return count;
}

// function size: useful instruction count
int AFPGAInliner::getFuncSize(Function *F) {
  // prepare tripcount based on pipeline unrolling
  std::map<BasicBlock*, int> BBMapCount;
#ifdef ESTIMATEION_BASED_PIPELINE
  PrepareInfoFromPipeline(F, BBMapCount);
#endif

  // count start from 1, "if only 1 ret instruction"
  int total_inst_number = 1;
  for (Function::iterator bb = F->begin(); bb != F->end(); bb++) {
    int bb_inst_number = 0;
    for (BasicBlock::iterator ii = bb->begin(); ii != bb->end(); ii++) {
      if (isa<CastInst>(ii)) continue;
      if (isa<ReturnInst>(ii)) continue;
      if (isa<DbgInfoIntrinsic>(ii)) continue;
      ++bb_inst_number;
    }

    BasicBlock *BB = &(*bb);
    int bb_count = (BBMapCount.count(BB) > 0) ? BBMapCount[BB] : 1;
    // pipeline exist, and cannot obtain tripcount:
    if (bb_count < 0) return -1;
    total_inst_number = total_inst_number + bb_inst_number * bb_count;
  }
  return total_inst_number;
}

// number of call to this function
int AFPGAInliner::getNumCallFunc(Function *callee, Function *caller, CallSite CS) {

  if(!callee || !caller || caller->isDeclaration() || callee->isDeclaration())
    return 0;

  int call_num = 0;
  for (auto &I : instructions(caller)) {
    CallSite callsite(cast<Value>(&I));
    if (!callsite || isa<IntrinsicInst>(I))
      continue;
    if (Function *current = callsite.getCalledFunction()) {
      if (current->isDeclaration())
        continue;
      // how many callee in current 'caller' function:
      if(current == callee)
        ++call_num;
    }
  }

#ifdef ESTIMATEION_BASED_PIPELINE
  // Consider callsite is in loops with pipeline
  CallInst* CI = dyn_cast<CallInst>(CS.getInstruction());
  if (!CI) return 0;
  int count_from_pipeline = upwardPipelineCount(caller, CI->getParent());
  if (count_from_pipeline < 0) {
    // callsite contained in pipeline loops, but we don't know tripcount
    return -1;
  }
  else if (count_from_pipeline == 0) {
    // no pipeline. Do nothing.
  }
  else {
    // update call_num. -1 means this callsite has been counted before
    call_num = (call_num - 1) + count_from_pipeline;
  }
#endif

  return call_num;
}

// If inline happens, how many increased instructions will be
int AFPGAInliner::increaseCodeSize(CallSiteInfo CSInfo) {
  // -1 means invalid values
  if (CSInfo.callee_size < 0 || CSInfo.call_num < 0)
    return -1;
  // normal:
  DEBUG(dbgs() << "callee size:[" << (CSInfo.callee_size)
      << "]   call num:[" << (CSInfo.call_num)
      << "]   total:[" << (CSInfo.callee_size) * (CSInfo.call_num)
      << "]\n");
  return (CSInfo.callee_size) * (CSInfo.call_num);
}


// the basic condition to do auto inline
bool AFPGAInliner::satisfyAutoInline(CallSiteInfo &CSInfo) {
  const bool doInline = true;
  const bool stopInline = false;

  // get caller & callee function
  CallSite CS = CSInfo.CS;
  Instruction* callInst = CS.getInstruction();
  CallInst* call = dyn_cast<CallInst>(callInst);
  if (!call) return stopInline;
  Function* callee = CS.getCalledFunction();
  Function* caller = callInst->getParent()->getParent();
  if (!caller || caller->isDeclaration()
    ||!callee || callee->isDeclaration())
    return stopInline;

  //1. If target function has only 1 ret: smallest callee function
  if (CSInfo.callee_size <= 1
   && CSInfo.callee_size >  0) {
    CSInfo.decision = MUST_INLINE;
    return doInline;
  }


#ifdef NEW_AUTOINLINE_FEATURE_20_1
  //2. resource increasing pattern: will not do inline
  //-  callee is called >1 times in same caller function
  //-  callee contains loop (no complete unroll)
  DominatorTree DT(*callee);
  LoopInfo LI(DT);
  if (CSInfo.call_num > 1) {
    for (auto *L : LI) {
      if (L != nullptr && LoopMayDisappear(L, DT, LI) == false)
        return stopInline;
    }
  }
#endif

  //3. Small callee function
  //   If increased size (deta function size) small enough
  int codechange = increaseCodeSize(CSInfo);
  if (codechange > 0 && codechange <= Threshold_deta_callee_size) {
    CSInfo.decision = MUST_INLINE;
    return doInline;
  }

#ifdef NEW_AUTOINLINE_FEATURE_20_1
  //4. If this callsite belongs to BB, which is init.check for guard variable
  //   This BB may be optimized in following auto-rom-infer.
  //   If callsite is inlined, it will destroy the IR pattern as auto-rom condition.
  //   So inlining will bring latency/II worse. Example case: QoR/suit1/viterbi_decoder
  //   CR:1049166
  if (isStaticGuardVarInitCheck(CS) == true)
    return stopInline;
#endif

#ifdef NEW_AUTOINLINE_FEATURE_20_1
  //5. callee_size + caller_size <= max threshold
  //-  corresponding to max scheduling time
  //-  If satisfy this condition, it NOT means will do inline.
  //-  Instead, it also depend on the 'sort order' later
  if (CSInfo.callee_size + CSInfo.caller_size <= Threshold_max_caller_size
   && CSInfo.callee_size > 0
   && CSInfo.caller_size > 0) {
    CSInfo.decision = NEED_CHECK_INLINE;
    return doInline;
  }
#endif

  return stopInline;
}




/////////////////////////////////////////////////////////
////     auto-rom-infer guard variable related       ////
////     check IR pattern for init.check             ////
/////////////////////////////////////////////////////////

static bool isGuardVariable(Value *V, std::string& GuardName) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    std::string GvName = getDemangleName(GV->getName());
    if (GvName.find("guard variable for") != std::string::npos) {
      GuardName = GvName;
      return true;
    }
  } else if (CastInst *CI = dyn_cast<CastInst>(V))
    return isGuardVariable(CI->getOperand(0), GuardName);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::BitCast)
      return isGuardVariable(CE->getOperand(0), GuardName);
  } else if (LoadInst *LDI = dyn_cast<LoadInst>(V))
    return isGuardVariable(LDI->getOperand(0), GuardName);

  return false;
}


BasicBlock *AFPGAInliner::getInitialBB(BranchInst *BI,
                                        GlobalVariable *&LocalStatic,
                                        std::string &GuardName) {
  auto *F = BI->getParent()->getParent();
  if (ICmpInst *IC = dyn_cast<ICmpInst>(BI->getCondition())) {
    Value *ICOp0 = IC->getOperand(0);
    ConstantInt *ICOp1 = dyn_cast<ConstantInt>(IC->getOperand(1));
    if (!isGuardVariable(ICOp0, GuardName) || !ICOp1)
      return nullptr;
    if (GlobalVariable *SGV = F->getParent()->getGlobalVariable(
      GuardName.substr(std::string("guard variable for ").size()), true)) {
      LocalStatic = SGV;
    }
    if (IC->getPredicate() == ICmpInst::ICMP_EQ)
      return  ICOp1->isZero() ? BI->getSuccessor(0) : BI->getSuccessor(1);
    else if (IC->getPredicate() == ICmpInst::ICMP_NE)
      return  ICOp1->isZero() ? BI->getSuccessor(1) : BI->getSuccessor(0);
    else
      return nullptr;
  } else {
    if (!isGuardVariable(BI->getCondition(), GuardName))
      return nullptr;
    if (GlobalVariable *SGV = F->getParent()->getGlobalVariable(
      GuardName.substr(std::string("guard variable for ").size()), true)) {
      LocalStatic = SGV;
    }
    return BI->getSuccessor(1);
  }
}

bool AFPGAInliner::isSpecailPatternAutoRomInitCheck(BasicBlock* InitBB, BranchInst *BI) {
  if (!InitBB || !BI)
    return false;
  TerminatorInst *TI = InitBB->getTerminator();
  if (TI && TI->getNumSuccessors() == 1) {
    BasicBlock *InitBBSucc = TI->getSuccessor(0);
    BasicBlock *BISucc0 = BI->getSuccessor(0);
    BasicBlock *BISucc1 = BI->getSuccessor(1);

    // pattern 1: Entry -> InitBB, Entry -> End, InitBB -> End
    if (InitBBSucc == BISucc0 || InitBBSucc == BISucc1)
      return true;

    // pattern 2: Entry -> InitBB, Entry -> ReadBB, InitBB -> End, ReadBB -> End
    TerminatorInst *TI0 = BISucc0->getTerminator();
    TerminatorInst *TI1 = BISucc1->getTerminator();
    if (TI0->getNumSuccessors() == 2 &&
        TI1->getNumSuccessors() == 1 &&
        TI0->getSuccessor(0) == TI1->getSuccessor(0))
      return true;

    BasicBlock *ReadBB = NULL;
    if (BISucc0 == InitBB) {
      ReadBB = BISucc1;
    } else if (BISucc1 == InitBB) {
      ReadBB = BISucc0;
    }
    if (ReadBB) {
      TerminatorInst *ReadBBTI = ReadBB->getTerminator();
      if (ReadBBTI &&
        ReadBBTI->getNumSuccessors() >= 1 &&
        ReadBBTI->getSuccessor(0) == InitBBSucc)
        return true;
    }
  }
  return false;
}

bool AFPGAInliner::shouldReserveForStaticInit(BasicBlock *BB) {
  std::string GuardName;
  GlobalVariable *LocalStatic;

  // 1. check if PreBB contains conditional branch
  if (!BB)
    return false;
  BasicBlock *PreBB = BB->getSinglePredecessor();
  if (!PreBB)
    return false;
  BranchInst *BI = dyn_cast<BranchInst>(PreBB->getTerminator());
  if (!BI || !BI->isConditional())
    return false;

  // 2. check if BB is InitialBB and match pattern
  if (getInitialBB(BI, LocalStatic, GuardName) == BB && isSpecailPatternAutoRomInitCheck(BB, BI))
    return true;

  return false;
}

bool AFPGAInliner::shouldReserveForStaticInit(Instruction *I) {
  if(!I)
    return false;
  return shouldReserveForStaticInit(I->getParent());
}

bool AFPGAInliner::shouldReserveForStaticInit(CallSite CS) {
  return shouldReserveForStaticInit(CS.getInstruction());
}

bool AFPGAInliner::isStaticGuardVarInitCheck(CallSite CS) {
  bool Res = shouldReserveForStaticInit(CS);
  if (Res)
    DEBUG(dbgs() << "BB:[" << CS.getInstruction()->getParent()->getName()
      << "] should reserve:" << Res << "\n");
  return Res;
}

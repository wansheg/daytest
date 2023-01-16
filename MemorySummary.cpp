
//
// This file implement the MemorySummary Analysis
// Memory Sumary is a ModulePass or CallGraphSCCPass that run on each function
// in post order in the call graph that :
// 1. Have a list of AliasSets.
// 2. Each alias set will contains the memory objects(Argument, AllocaInst,
// GlobalVariable)[1] that(may) alias each other[2], and the actual pointers
// that are accessed(gep, bitcast)
// 2.a. Put the bitcast to the AliasSet as well
// 3. At the same time we will have a pointer - to - access map, that map the
// pointers to the actual accesses(load, store and call)
// 4. Based on the pointer-to-access map, we know the actual accessed
// type(access width) and alignment(suppose we run the AlignMemory pass)
// 4.a. Also record the "structs path" in the geps
// [1]Replace alloca by static global may simplify the problem ? -No before we
// duplicate the function to dis - alias objects
// [2]Notice that pointer alias is not transitive : C alias A and B alias C do
// not imply A alias B.
//
// With these information, the MemorySumary pass can tell that :
// 1. What are the objects that alias each other, which need to be assign to the
// same memory bank(memory address space)
// 2. For each memory bank(AliasSet), what are the accessed types and the
// minimal
// alignments
// 3. What are the memory transform attributes(array_partition, array_map,
// array_reshape) applied to the AliasSet ?
// 3.a. Do they conflict ?
//
// Based on this information, we can implement BRAM for each memory bank:
// 1. Based on the size, alignment of objects in the memory bank, we can
// calculate
// the size of the memory bank in bytes.
// 2. Based on the minimal alignment and minimal accessed type we can decide the
// word width of the memory bank(suppose we do not have byteenable)
// 2.a. With byteenable we can have memory bank that have a bigger word width
// than
// size of the minimal type
// 2.b. We can also use read - modify - write to emulate byteenable
// 2.c. We can also emulate byteenable by array partition with if statement
// 3. Based on the set of accessed types we can decide whether we can eliminate
// the
// implicit paddings
// 4. Each memory bank is represented by a 1 - D array, so we need to linearize
// the
// addresses
//
// Please notice we need to do Unified Array Transform when we materialize the
// memory banks
//
// When we materialize the memory access, we should also annotate the byte
// offset and the access width to improve loop - dependency analysis
//
// Notice that we can vectorize the internal memory access, in the worst case we
// just need to sequentialize the vector access to undo the vectorization
//
//===----------------------------------------------------------------------===//

#include "afgpa/TransformUtils/MiscUtil.h"
#include "afgpa/Memory/MemorySummary.h"
#include "llvm/IR/XILINXFPGAIntrinsicInst.h"
#include "afgpa/LinkAllPasses.h"
#include "afgpa/Memory/afgpaAA.h"
#include "afgpa/Options.h"
#include "afgpa/afgpaConfig.h"
#include "afgpa/SPIR/IntrinsicInst.h"
#include "afgpa/Support/ArrayExtras.h"
#include "llvm/IR/XILINXAggregateUtil.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
//#include "llvm/Analysis/ObjCARCAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DerivedUser.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "afgpa-memory-summary"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <unordered_set>
using namespace llvm;
using namespace PatternMatch;

static bool IsInterfaceObject(const Value *V) {
  return isa<Argument>(V) || isa<GlobalVariable>(V) ||
         match(V, m_Intrinsic<Intrinsic::fpga_get_printf_buffer>());
}

static bool IsIdentifiedObject(const Value *V) {
  return isa<AllocaInst>(V) || IsInterfaceObject(V);
}

/// This is the function that does the work of looking through basic
/// ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant, a multiplied value, or a phi, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul &&
           !isa<PHINode>(U->getOperand(1))))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (true);
}

namespace llvm {
class ReturnAddr : public DerivedUser {
  static void deleteMe(DerivedUser *Self) {
    delete static_cast<DerivedUser *>(Self);
  }

  void *operator new(size_t s) { return User::operator new(s); }

  ReturnAddr(LLVMContext &C)
      : DerivedUser(Type::getInt8PtrTy(C), MemoryDefVal, nullptr, 0, deleteMe) {
  }

  friend class ArgumentNoAlias;

public:
  static bool classof(const Value *V) {
    return V->getValueID() == MemoryDefVal;
  }
};

typedef MemorySummaryWrapperPass::GlobalVariableEC GlobalVariableEC;
class ArgumentNoAlias : public AAResultBase<ArgumentNoAlias> {
  const DataLayout &DL;
  const GlobalVariableEC &GEC;
  MemorySummary &MS;
  std::unique_ptr<ReturnAddr> Ret;
  std::unordered_set<const Value *> Returned;
  DenseMap<const Value *, SmallVector<Value*, 4>> UnderlyingObjects;
public:
  ArgumentNoAlias(const DataLayout &DL, const GlobalVariableEC &GEC,
                  MemorySummary &MS)
      : AAResultBase(), DL(DL), GEC(GEC), MS(MS) {}
  ArgumentNoAlias(ArgumentNoAlias &&Arg)
      : AAResultBase(std::move(Arg)), DL(Arg.DL), GEC(Arg.GEC), MS(Arg.MS),
        Ret(std::move(Arg.Ret)), Returned(std::move(Arg.Returned)), UnderlyingObjects(std::move(Arg.UnderlyingObjects)) {}
  ~ArgumentNoAlias() {}

  ReturnAddr *getReturnAddr() const { return Ret.get(); }
  void addReturnedValue(const Value *V) {
    if (!getReturnAddr())
      Ret.reset(new ReturnAddr(V->getContext()));

    SmallVector<Value *, 4> Objs;
    findUnderlyingObjects(V, Objs);
    Returned.insert(Objs.begin(), Objs.end());
  }

  bool canIdentifyReturns() const {
    return llvm::all_of(Returned, IsInterfaceObject);
  }

  const Value *getAliasSetRepresentative(const Value *V) const {
    if (auto *G = dyn_cast<GlobalVariable>(V))
      return GEC.getLeaderValue(G);

    return V;
  }

  bool findReturnedObjects(const CallInst *CI, SmallVectorImpl<Value *> &Objs) {
    auto *Callee = CI->getCalledFunction();
    if (!Callee || Callee->isDeclaration())
      return false;

    auto *Summary = MS.getSummary(Callee);
    auto *Bank = Summary->getReturnBank();

    if (!Bank)
      return false;

    for (Value *O : Bank->objects()) {
      if (isa<Constant>(O)) {
        getUnderlyingOrReturnedObjects(O, Objs);
        continue;
      }

      // Need to translate formal parameters to actual parameters
      if (auto *A = dyn_cast<Argument>(O)) {
        getUnderlyingOrReturnedObjects(CI->getArgOperand(A->getArgNo()), Objs);
        continue;
      }

      // Alloca instructions are local to the function call and can't be
      // live-out, we might see them due to AA being approximate.
      if (isa<AllocaInst>(O)) {
        continue;
      }

      llvm_unreachable("Unexpected object!");
    }

    return true;
  }

  void getUnderlyingOrReturnedObjects(Value *O,
                                      SmallVectorImpl<Value *> &Objs) {
    if (auto *CI = dyn_cast<CallInst>(O)) {
      if (auto *RV = CI->getReturnedArgOperand())
        O = RV;
      else if (findReturnedObjects(CI, Objs))
        return;
    }

    SmallVector<Value *, 4> Ptrs;
    GetUnderlyingObjects(O, Ptrs, DL, nullptr,
                         afgpaConfig::GlobalConfig().AAMaxSearchDepth);
    for (auto *P : Ptrs) {
      if (P == O) {
        Objs.push_back(P);
        continue;
      }

      if (auto *CI = dyn_cast<CallInst>(P)) {
        getUnderlyingOrReturnedObjects(CI, Objs);
        continue;
      }

      Objs.push_back(P);
    }
  }

  void findUnderlyingObjects(const Value *V, SmallVectorImpl<Value *> &Objects);

  AliasResult aliasUnderlyingObject(const Value *A, const Value *B) {
    auto *O0 = getAliasSetRepresentative(A), *O1 = getAliasSetRepresentative(B);
    if (O0 == O1)
      return MustAlias;
    // If A/B point to two different objects, we know that we have no alias.
    if (isIdentifiedObject(O0) && isIdentifiedObject(O1))
      return NoAlias;
    // Assume interface objects do not alias
    // Need more explaination
    bool IsO0Interface = IsInterfaceObject(O0);
    bool IsO1Interface = IsInterfaceObject(O1);
    if (IsO0Interface && IsO1Interface)
      return NoAlias;

    // Alloca do not alias with arguments
    if (IsO0Interface && isa<AllocaInst>(O1))
      return NoAlias;
    if (isa<AllocaInst>(O0) && IsO1Interface)
      return NoAlias;

    // Assume other pointers are not null
    if (isa<ConstantPointerNull>(O0) || isa<ConstantPointerNull>(O1))
      return NoAlias;

    // Check alias on return value
    bool IsO0Function = O0 == Ret.get();
    bool IsO1Function = O1 == Ret.get();
    if (IsO0Function && IsO1Interface)
      return Returned.count(O1) ? MustAlias : NoAlias;
    if (IsO0Interface && IsO1Function)
      return Returned.count(O0) ? MustAlias : NoAlias;

    return AAResultBase::alias(MemoryLocation(O0), MemoryLocation(O1));
  }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    auto APT = cast<PointerType>(LocA.Ptr->getType());
    auto BPT = cast<PointerType>(LocB.Ptr->getType());

    // Pointer from different address space do not alias
    if (APT->getAddressSpace() != BPT->getAddressSpace())
      return NoAlias;

    SmallVector<Value *, 4> OA, OB;
    findUnderlyingObjects(LocA.Ptr, OA);
    findUnderlyingObjects(LocB.Ptr, OB);
    assert(!OA.empty() && !OB.empty());

    bool AnyAlias = false, AllMustAlias = true;
    for (auto *A : OA)
      for (auto *B : OB) {
        auto R = aliasUnderlyingObject(A, B);
        AnyAlias |= R != NoAlias;
        AllMustAlias &= R == MustAlias;
      }

    if (!AnyAlias)
      return NoAlias;

    if (AllMustAlias)
      return MustAlias;

    return MayAlias;
  }
};
} // namespace llvm

void ArgumentNoAlias::findUnderlyingObjects(const Value *V,
                                            SmallVectorImpl<Value *> &Objects) {
  auto res = UnderlyingObjects.find(V);
  if (res != UnderlyingObjects.end()) {
    Objects.assign(res->second.begin(), res->second.end());
    return;
  }

  SmallPtrSet<const Value *, 16> Visited;
  SetVector<Value *> ObjSet;
  SmallVector<const Value *, 4> Working(1, V);
  Visited.insert(V);

  do {
    const Value *PtrV = Working.pop_back_val();
    SmallVector<Value *, 4> Objs;
    getUnderlyingOrReturnedObjects(const_cast<Value *>(PtrV), Objs);

    for (Value *O : Objs) {

      if (Operator::getOpcode(O) == Instruction::IntToPtr) {
        const Value *IntPtr =
            getUnderlyingObjectFromInt(cast<User>(O)->getOperand(0));
        if (IntPtr->getType()->isPointerTy() && Visited.insert(IntPtr).second) {
          Working.push_back(IntPtr);
          continue;
        }
      }

      // If GetUnderlyingObjects fails to find an identifiable object,
      // getUnderlyingObjectsForCodeGen also fails for safety.
      if (!IsIdentifiedObject(O)) {
        Objects.clear();
        auto res = const_cast<Value *>(GetUnderlyingObject(V, DL));
        Objects.push_back(res);
        UnderlyingObjects[V].push_back(res);
        return;
      }

      ObjSet.insert(O);
    }
  } while (!Working.empty());

  UnderlyingObjects[V].assign(ObjSet.begin(), ObjSet.end());
  Objects.assign(ObjSet.begin(), ObjSet.end());
}

namespace llvm {
static uint64_t CalculateTypeSizeInBits(const DataLayout &DL, Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return CalculateTypeSizeInBits(DL, AT->getElementType());

  // Do not fail on pointer to function or opaque struct.
  // Assume function and opaque struct has type size of 1 byte
  if (!T->isSized())
    return 8;

  if (auto *ST = dyn_cast<StructType>(T)) {
    if (DL.getTypeAllocSizeInBits(ST) >
        afgpaConfig::GlobalConfig().MaxTypeSizeThreshold)
      return std::accumulate(ST->element_begin(), ST->element_end(),
                             uint64_t(8), [&DL](uint64_t Max, Type *T) {
                               uint64_t Size = CalculateTypeSizeInBits(DL, T);
                               return std::max(Size, Max);
                             });
    if (ST->getStructNumElements() == 1)
      return CalculateTypeSizeInBits(DL, ST->getStructElementType(0));
  }
  return DL.getTypeSizeInBits(T);
}

static uint64_t CalculateTypeAllocSizeInBits(const DataLayout &DL, Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return CalculateTypeAllocSizeInBits(DL, AT->getElementType());

  // Do not fail on pointer to function or opaque struct.
  // Assume function and opaque struct has type size of 1 byte
  if (!T->isSized())
    return 8;

  if (auto *ST = dyn_cast<StructType>(T))
    if (DL.getTypeAllocSizeInBits(ST) >
        afgpaConfig::GlobalConfig().MaxTypeSizeThreshold)
      return std::accumulate(ST->element_begin(), ST->element_end(),
                             uint64_t(8), [&DL](uint64_t Max, Type *T) {
                               uint64_t Size = CalculateTypeAllocSizeInBits(DL, T);
                               return std::max(Size, Max);
                             });

  return DL.getTypeAllocSizeInBits(T);
}

static uint64_t CalculateTypeStoreSizeInBits(const DataLayout &DL, Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return CalculateTypeStoreSizeInBits(DL, AT->getElementType());

  // Do not fail on pointer to function or opaque struct.
  // Assume function and opaque struct has type size of 1 byte
  if (!T->isSized())
    return 8;

  if (auto *ST = dyn_cast<StructType>(T))
    if (DL.getTypeStoreSizeInBits(ST) >
        afgpaConfig::GlobalConfig().BRAMMaxDataSizeInBits)
      return std::accumulate(ST->element_begin(), ST->element_end(),
                             uint64_t(8), [&DL](uint64_t Min, Type *T) {
                               uint64_t Size =
                                   CalculateTypeStoreSizeInBits(DL, T);
                               return std::min(Size, Min);
                             });

  return DL.getTypeStoreSizeInBits(T);
}

struct AccessInfo {
  Instruction *I;
  bool IsStream;
  bool IsMAXI;
  uint64_t SeqReadSizeInBytes, SeqWriteSizeInBytes;
  uint64_t MinAccessSizeInByte, MaxAccessSizeInByte;

  static AccessInfo maxi(MAXIStoreInst &I, const DataLayout &DL) {
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I, false, true, 0, 0, EltSize, EltSize};
  }
  static AccessInfo maxi(MAXIIOInst &I, const DataLayout &DL) {
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I, false, true, 0, 0, EltSize, EltSize};
  }

  static AccessInfo stream(spir::PipeBlockingReadInst &I,
                           const DataLayout &DL) {
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I, true, false, 0, 0, EltSize, EltSize};
  }

  static AccessInfo stream(spir::PipeBlockingWriteInst &I,
                           const DataLayout &DL) {
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I, true, false, 0, 0, EltSize, EltSize};
  }

  static AccessInfo stream(FPGAFIFOInst &I, const DataLayout &DL) {
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I, true, false, 0, 0, EltSize, EltSize};
  }

  static AccessInfo stream(AXISIntrinsicInst &I, uint8_t ChID,
                           const DataLayout &DL) {
    auto *Ch = I.getArgOperand(ChID);
    auto *Obj = GetUnderlyingObject(Ch, DL);
    auto *EltTy = Obj->getType()->getPointerElementType();
    auto EltSize = DL.getTypeAllocSize(EltTy);
    return {&I, true, false, 0, 0, EltSize, EltSize};
  }

  static AccessInfo unknown(Instruction &I, uint64_t AccessSizeInByte) {
    return {&I, false, false, 0, 0, AccessSizeInByte, AccessSizeInByte};
  }

  static AccessInfo seq(SeqBeginInst &I, const DataLayout &DL) {
    uint64_t SeqReadSizeInBytes = 0, SeqWriteSizeInBytes = 0;
    if (IsIdentifiedObject(I.getPointerOperand()->stripPointerCasts())) {
      auto Size = I.getSmallConstantSizeInBytes(DL);
      if (Size == 0)
        Size = MemoryLocation::UnknownSize;
      if (I.isLoad())
        SeqReadSizeInBytes = Size;
      else
        SeqWriteSizeInBytes = Size;
    }
    auto DataT = I.getDataType();
    auto EltSize = DL.getTypeAllocSize(DataT);
    return {&I,      false,  false, SeqReadSizeInBytes, SeqWriteSizeInBytes,
            EltSize, EltSize};
  }

  static AccessInfo propagate(Instruction &I, Value *Param,
                              const MemoryBank &Bank) {
    uint64_t SeqReadSizeInBytes = 0, SeqWriteSizeInBytes = 0;
    if (!Bank.isStream() && IsIdentifiedObject(Param->stripPointerCasts())) {
      SeqReadSizeInBytes = Bank.getSeqReadSizeInBytes();
      SeqWriteSizeInBytes = Bank.getSeqWriteSizeInBytes();
    }

    return {&I,
            Bank.isStream(),
            Bank.isMAXI(),
            SeqReadSizeInBytes,
            SeqWriteSizeInBytes,
            Bank.getMinSize(),
            Bank.getMaxSize()};
  }
};

struct MemorySummaryBuilder : public InstVisitor<MemorySummaryBuilder> {
  MemorySummary &MS;
  FunctionMemorySummary CurSummary;
  Function &F;
  AliasSetTracker AST;
  FunctionMemorySummary::IntrinsicSet Intrinsics;
  const DataLayout DL;
  typedef DenseMap<Value *, unsigned> AlignMap;
  AlignMap ReadAligns, WriteAligns;
  std::set<Value *> BitCasts;
  std::set<Value *> ShiftRegs;
  const bool AssumeAlignAccesses;
  ArgumentNoAlias &ANA;
  DenseMap<Value *, SmallVector<AccessInfo, 4>> PtrToInsts;

  unsigned guessAlign(Value *V) const {
    if (AssumeAlignAccesses)
      return DL.getPrefTypeAlignment(V->getType()->getPointerElementType());

    return 1;
  }

  unsigned guessAlign(Value *V, unsigned Align) const {
    if (Align == 0)
      return guessAlign(V);

    return Align;
  }

  MemorySummaryBuilder(MemorySummary &MS, Function &F, AliasAnalysis &AA,
                       ArgumentNoAlias &ANA)
      : MS(MS), F(F), AST(AA, false), DL(F.getParent()->getDataLayout()),
        AssumeAlignAccesses(afgpaConfig::GlobalConfig().AssumeAlignAccesses),
        ANA(ANA) {}

  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }

  void run();

  void addAlignment(AlignMap &Alignments, Value *V, unsigned Alignment) {
    if (Alignment == 0)
      Alignment = guessAlign(V);

    auto P = Alignments.insert({V, Alignment});
    if (!P.second)
      P.first->second = GreatestCommonDivisor64(P.first->second, Alignment);
  }

  bool hasBitcast(const AliasSet &AS) const;
  bool isShiftReg(const AliasSet &AS) const;
  void analyzeAccessPattern(MemoryBank &Bank, ArrayRef<AccessInfo> Insts);
  bool isReturned(const AliasSet &AS) const;

  MemoryBank createMemoryBank(const AliasSet &AS);

  void visitBitCastInst(BitCastInst &I);
  void visitPHINode(PHINode &I);

  void visitGetElementPtrInst(GetElementPtrInst &I);

  void visitSelectInst(SelectInst &I);

  void visitLoadInst(LoadInst &I);

  void visitStoreInst(StoreInst &I);

  void visitPPPOLoadInst(FPGAPPPOLoadInst &I);
  void visitPPPOStoreInst(FPGAPPPOStoreInst &I);

  void visitAllocaInst(AllocaInst &I);

  void visitReturnInst(ReturnInst &I);

  void visitCallInst(CallInst &I);
  void visitIntrinsicInst(IntrinsicInst &I);

  void visitSeqBeginInst(SeqBeginInst &I);
  void visitFPGAFIFOInst(FPGAFIFOInst &I);
  void visitAXISIntrinsic(AXISIntrinsicInst &I);
  void visitPipeBlockingReadInst(spir::PipeBlockingReadInst &I);
  void visitPipeBlockingWriteInst(spir::PipeBlockingWriteInst &I);
  void visitShiftRegInst(ShiftRegInst &I);
  void visitMAXIIOInst(MAXIIOInst &I);
  void visitMAXIStoreInst(MAXIStoreInst &I);
  void visitSideeffect(IntrinsicInst &I);
  void visitAggregateInst(AggregateInst &I);

  void visitMemTransferInst(MemTransferInst &I);
  void visitMemSetInst(MemSetInst &I);

  void visitSSACopy(SSACopyInst &I);
};
} // namespace llvm

bool MemoryBank::IsObject(Value *V) {
  return isa<Argument>(V) || isa<AllocaInst>(V) || isa<GlobalVariable>(V);
}

MemoryBank::MemoryBank(const AliasSet &Set, bool HasBitCast,
                       bool IsArrayOfShiftReg)
    : MinReadAlign(0), MinWriteAlign(0), MaxAlign(0), MaxSize(0), MinSize(0),
      MinTypeAlign(0), MaxTypeSize(0), PreciseTypeSizeInBits(0),
      MinTypeSizeFromStore(0), AS(0), HasBitCast(HasBitCast), IsStream(false),
      IsMAXI(false), IsShiftReg(IsArrayOfShiftReg), IsOnlyOneField(true),
      SeqReadSizeInBytes(0), SeqWriteSizeInBytes(0), AggrInfo(AggrInfo) {
  for (auto P :
       make_filter_range(Set, [](auto P) { return IsObject(P.getValue()); }))
    Objs.push_back(P.getValue());

  assert(!Objs.empty() && "Unexpected empty object!");
  AS = Objs.front()->getType()->getPointerAddressSpace();
  assert(llvm::all_of(Objs,
                      [this](auto V) {
                        return AS == V->getType()->getPointerAddressSpace();
                      }) &&
         "Pointers from different address space do not alias!");
}

void MemoryBank::print(raw_ostream &OS) const {
  OS << "Bank AS:" << AS << " [";
  for (Value *O : objects())
    OS << *O->getType() << ' ' << O->getName() << ", ";
  OS << "]\n";
  OS.indent(2) << "MinReadAlign:" << MinReadAlign << '\n';
  OS.indent(2) << "MinWriteAlign:" << MinWriteAlign << '\n';
  OS.indent(2) << "MaxAlign:" << MaxAlign << '\n';
  OS.indent(2) << "MaxSize:" << MaxSize << '\n';
  OS.indent(2) << "MinSize:" << MinSize << '\n';
  OS.indent(2) << "MinTypeAlign:" << MinTypeAlign << '\n';
  OS.indent(2) << "MaxTypeSize:" << MaxTypeSize << '\n';
  OS.indent(2) << "PreciseTypeSizeInBits:" << PreciseTypeSizeInBits << '\n';
  OS.indent(2) << "MinTypeSizeFromStore:" << MinTypeSizeFromStore << '\n';
  OS.indent(2) << "IsShiftReg:" << IsShiftReg << '\n';
  OS.indent(2) << "HasBitCast:" << HasBitCast << '\n';
  OS.indent(2) << "IsStream:" << IsStream << '\n';
  OS.indent(2) << "IsMAXI:" << IsMAXI << '\n';
  OS.indent(2) << "SeqReadSizeInBytes:" << SeqReadSizeInBytes << '\n';
  OS.indent(2) << "SeqWriteSizeInBytes:" << SeqWriteSizeInBytes << '\n';
  OS.indent(4) << AggrInfo.WordSize << ". // 0 means no spec" << '\n';
  // aggregate info -- begin
  OS.indent(2) << "Instructions:" << '\n';
  for (auto I : Insts) {
    OS.indent(2) << *I << '\n';
  }
}



LLVM_DUMP_METHOD void MemoryBank::dump() const { print(dbgs()); }

static uint64_t CalculateElementSizeInBits(const DataLayout &DL, Type *T) {
  if (auto *AT = dyn_cast<ArrayType>(T))
    return CalculateElementSizeInBits(DL, AT->getElementType());

  if (auto *ST = dyn_cast<StructType>(T)) {
    return std::accumulate(ST->element_begin(), ST->element_end(), uint64_t(8),
                           [&DL](uint64_t Max, Type *T) {
                             uint64_t Size = CalculateElementSizeInBits(DL, T);
                             return std::max(Size, Max);
                           });
  }

  // Do not fail on pointer to function.
  // Assume function has an element size of 1 byte
  if (isa<FunctionType>(T))
    return 8;

  return DL.getTypeAllocSizeInBits(T);
}

void MemoryBank::initMaxAlign(const DataLayout &DL) {
  for (Value *Ptr : Objs) {
    auto *T = Ptr->getType()->getPointerElementType();
    MaxAlign = std::max(MaxAlign, CalculateElementSizeInBits(DL, T) / 8);
  }
}

void MemoryBank::initMinTypeAlign(const DataLayout &DL) {
  for (Value *Ptr : Objs) {
    auto *T = Ptr->getType()->getPointerElementType();
    unsigned TypeAlign = T->isSized() ? DL.getABITypeAlignment(T) : 1u;
    MinTypeAlign = GreatestCommonDivisor64(MinTypeAlign, TypeAlign);
  }
}

void MemoryBank::initMaxTypeSize(const DataLayout &DL) {
  for (Value *Ptr : Objs) {
    auto *T = Ptr->getType()->getPointerElementType();
    MaxTypeSize = std::max(MaxTypeSize, CalculateTypeAllocSizeInBits(DL, T) / 8);
  }
}

void MemoryBank::initPreciseTypeSize(const DataLayout &DL) {
  for (Value *Ptr : Objs) {
    auto *T = Ptr->getType()->getPointerElementType();
    PreciseTypeSizeInBits = std::max(PreciseTypeSizeInBits, CalculateTypeSizeInBits(DL, T));
  }
}

static Type *stripArrayType(Type *Ty) {
  if (Ty->isArrayTy())
    return stripArrayType(Ty->getArrayElementType());
  return Ty;
}

void MemoryBank::initOnlyOneField() {
  if (hasBitcast()) {
    IsOnlyOneField = false;
    return;
  }
  for (Value *Ptr : Objs) {
    auto *T = Ptr->getType()->getPointerElementType();
    IsOnlyOneField &= afgpa::IsScalarTyOrStructTyWithOnlyOneField(stripArrayType(T));
  }
}

void MemoryBank::updateMinTypeSizeFromStore(Value *Ptr, const DataLayout &DL) {
  auto *T = Ptr->getType()->getPointerElementType();
  MinTypeSizeFromStore = GreatestCommonDivisor64(
      MinTypeSizeFromStore, CalculateTypeStoreSizeInBits(DL, T) / 8);
}

void MemoryBank::addReadAlign(uint64_t Algin) {
  MaxAlign = LeastCommonMultiple64(MaxAlign, Algin);
  MinReadAlign = GreatestCommonDivisor64(MinReadAlign, Algin);
}

void MemoryBank::addWriteAlign(uint64_t Algin) {
  MaxAlign = LeastCommonMultiple64(MaxAlign, Algin);
  MinWriteAlign = GreatestCommonDivisor64(MinWriteAlign, Algin);
}

static bool IsPointerBitCast(BitCastOperator &BC) {
  if (!BC.getType()->isPointerTy())
    return false;

  if (onlyUsedByLifetimeMarkers(&BC))
    return false;

  return true;
}

static bool HasConstantBitCast(ConstantExpr *CE) {
  switch (CE->getOpcode()) {
  case Instruction::BitCast: {
    auto *Cast = cast<BitCastOperator>(CE);
    if (IsPointerBitCast(*Cast))
      return true;

    auto *Ptr = Cast->getOperand(0);
    if (auto *C = dyn_cast<ConstantExpr>(Ptr))
      return HasConstantBitCast(C);
    return false;
  }
  case Instruction::GetElementPtr: {
    auto *Ptr = cast<GEPOperator>(CE)->getPointerOperand();
    if (auto *C = dyn_cast<ConstantExpr>(Ptr))
      return HasConstantBitCast(C);
    return false;
  }
  default:
    return false;
  }
}

static bool ImplyBitCastFromType(Type *T) {
  if (T->isPointerTy())
    return true;

  if (T->isVectorTy())
    return true;

  if (auto *AT = dyn_cast<ArrayType>(T))
    return ImplyBitCastFromType(AT->getElementType());

  if (auto *ST = dyn_cast<StructType>(T))
    return llvm::any_of(ST->elements(), ImplyBitCastFromType);

  return false;
}

static bool ImplyBitCast(Value *V) {
  if (auto *C = dyn_cast<ConstantExpr>(V))
    return HasConstantBitCast(C);

  if (auto *GEP = dyn_cast<GetElementPtrInst>(V))
    return ImplyBitCast(GEP->getPointerOperand());

  // Returning from function also imply bitcast
  if (isa<CallInst>(V))
    return true;

  return ImplyBitCastFromType(V->getType()->getPointerElementType());
}

bool MemorySummaryBuilder::hasBitcast(const AliasSet &AS) const {
  return llvm::any_of(AS, [this](auto P) {
    auto *Ptr = P.getValue();
    return BitCasts.count(Ptr) || ImplyBitCast(Ptr);
  });
}

bool MemorySummaryBuilder::isShiftReg(const AliasSet &AS) const {
  return llvm::any_of(AS, [this](auto P) {
    auto *Ptr = P.getValue();
    return ShiftRegs.count(Ptr);
  });
}

bool MemorySummaryBuilder::isReturned(const AliasSet &AS) const {
  return llvm::any_of(AS, [](auto P) { return isa<ReturnAddr>(P.getValue()); });
}

void MemorySummaryBuilder::analyzeAccessPattern(MemoryBank &Bank,
                                                ArrayRef<AccessInfo> Accesses) {
  SmallPtrSet<Instruction *, 8> UniqueInsts;
  uint64_t SeqReadSizeInBytes = 0, SeqWriteSizeInBytes = 0;
  unsigned NumSeqRead = 0, NumSeqWrite = 0, NumOthers = 0;
  for (const auto &A : Accesses) {
    Bank.addAccessSize(A.MinAccessSizeInByte);
    Bank.addAccessSize(A.MaxAccessSizeInByte);

    if (!UniqueInsts.insert(A.I).second)
      continue;

    Bank.Insts.push_back(A.I);
    if (A.SeqReadSizeInBytes) {
      SeqReadSizeInBytes += A.SeqReadSizeInBytes;
      ++NumSeqRead;
    }

    if (A.SeqWriteSizeInBytes) {
      SeqWriteSizeInBytes += A.SeqWriteSizeInBytes;
      ++NumSeqWrite;
    }

    if (!A.SeqReadSizeInBytes && !A.SeqWriteSizeInBytes)
      ++NumOthers;
  }

  if (Accesses.empty())
    return;

  Bank.IsStream =
      llvm::all_of(Accesses, [](AccessInfo Info) { return Info.IsStream; });

  Bank.IsMAXI =
      llvm::all_of(Accesses, [](AccessInfo Info) { return Info.IsMAXI; });

  if (Bank.IsStream || Bank.IsMAXI)
    return;

  // Sequential access pattern
  if (NumOthers == 0 && NumSeqRead <= 1 && NumSeqWrite <= 1) {
    Bank.SeqReadSizeInBytes = SeqReadSizeInBytes;
    Bank.SeqWriteSizeInBytes = SeqWriteSizeInBytes;
  }
}

MemoryBank MemorySummaryBuilder::createMemoryBank(const AliasSet &AS) {
  MemoryBank B(AS, hasBitcast(AS), isShiftReg(AS));
  SmallVector<AccessInfo, 8> Insts;

  B.initMaxAlign(DL);

  B.initMinTypeAlign(DL);
  B.initMaxTypeSize(DL);
  B.initPreciseTypeSize(DL);
  B.initOnlyOneField();

  unsigned L = MemoryBank::NoAccess;
  for (auto V : AS) {
    auto *Ptr = V.getValue();
    if (auto A = ReadAligns.lookup(Ptr)) {
      B.addReadAlign(A);
      L |= MemoryBank::RefAccess;
    }

    if (auto A = WriteAligns.lookup(Ptr)) {
      B.addWriteAlign(A);
      B.updateMinTypeSizeFromStore(Ptr, DL);
      L |= MemoryBank::ModAccess;
    }

    auto I = PtrToInsts.find(Ptr);
    if (I != PtrToInsts.end())
      Insts.append(I->second.begin(), I->second.end());
  }

  analyzeAccessPattern(B, Insts);

  // Update the access summary for global variable
  for (Value *V : B.objects())
    if (auto *G = dyn_cast<GlobalVariable>(V))
      MS.addGlobalAccess(G, L);

  B.setAccessMode(L);

  DEBUG(dbgs() << "\nNew memory bank:\n"; B.dump());
  return B;
}

static bool IsPointerArray(Type *T) {
  if (T->isPointerTy())
    return true;

  if (auto *AT = dyn_cast<ArrayType>(T))
    return IsPointerArray(AT->getElementType());

  return false;
}

static void ExtractUnderlyingObject(Constant *C, const DataLayout &DL,
                                    SmallVectorImpl<GlobalVariable *> &Objs) {
  if (auto *A = dyn_cast<ConstantArray>(C)) {
    for (auto *E : A->operand_values())
      ExtractUnderlyingObject(cast<Constant>(E), DL, Objs);
    return;
  }

  auto *O = GetUnderlyingObject(C, DL);
  if (auto *G = dyn_cast<GlobalVariable>(O))
    Objs.push_back(G);
}

static bool ExtractUnderlyingObject(GlobalVariable &G, const DataLayout &DL,
                                    SmallVectorImpl<GlobalVariable *> &Objs) {
  assert(Objs.empty() && "Expect empty vector!");
  if (!G.hasInitializer())
    return false;

  auto *I = G.getInitializer();
  if (!IsPointerArray(I->getType()))
    return false;

  ExtractUnderlyingObject(I, DL, Objs);
  return !Objs.empty();
}

void MemorySummaryBuilder::print(raw_ostream &OS) const {
  OS << "Memory Summary of Function: " << F.getName() << '\n';
  AST.print(OS);
  OS << "Read alignments:\n";
  for (auto P : ReadAligns)
    OS.indent(2) << *P.first << '[' << P.second << "]\n";
  OS << "Write alignments:\n";
  for (auto P : WriteAligns)
    OS.indent(2) << *P.first << '[' << P.second << "]\n";
}

void MemorySummaryBuilder::run() {
  for (auto &A : F.args()) {
    if (!A.getType()->isPointerTy())
      continue;

    AST.add(&A, MemoryLocation::UnknownSize, AAMDNodes());
    if (auto Size = guessArrayDecayedDimSize(&A)) {
      auto *EltT = A.getType()->getPointerElementType();
      CurSummary.addArrayGeometry(&A, ArrayType::get(EltT, Size));
    }
  }

  visit(&F);

  SmallVector<GlobalVariable *, 8> Objs;
  for (auto &GV : F.getParent()->globals()) {
    if (AST.getAliasSetForPointerIfExists(&GV, MemoryLocation::UnknownSize,
                                          AAMDNodes())) {
      AST.add(&GV, MemoryLocation::UnknownSize, AAMDNodes());
      // Generate the alias-set for the look-up table as well
      if (ExtractUnderlyingObject(GV, DL, Objs)) {
        for (auto *O : Objs)
          AST.add(O, MemoryLocation::UnknownSize, AAMDNodes());
        Objs.clear();
      }
    }
  }

  // Force alias of all returned objects
  if (auto *Ret = ANA.getReturnAddr()) {
    if (ANA.canIdentifyReturns()) {
      AST.add(Ret, MemoryLocation::UnknownSize, AAMDNodes());
    }
  }

  CurSummary.addIntrinsics(Intrinsics);

  for (const auto &AS : AST) {
    if (AS.isForwardingAliasSet())
      continue;

    if (!llvm::any_of(
            AS, [](auto P) { return MemoryBank::IsObject(P.getValue()); }))
      continue;

    auto Bank = createMemoryBank(AS);
    CurSummary.addMemoryBank(std::move(Bank), isReturned(AS));
  }

  MS.addSummary(&F, std::move(CurSummary));
}

void MemorySummaryBuilder::visitBitCastInst(BitCastInst &I) {
  if (!IsPointerBitCast(cast<BitCastOperator>(I)))
    return;

  AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
  BitCasts.insert(&I);
}

void MemorySummaryBuilder::visitPHINode(PHINode &I) {
  if (!I.getType()->isPointerTy())
    return;

  AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
  BitCasts.insert(&I);
}


void MemorySummaryBuilder::visitSelectInst(SelectInst &I) {
  if (!I.getType()->isPointerTy())
    return;

  AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
}

void MemorySummaryBuilder::visitGetElementPtrInst(GetElementPtrInst &I) {
  AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
}

void MemorySummaryBuilder::visitReturnInst(ReturnInst &I) {
  auto *RV = I.getReturnValue();
  if (!RV || !RV->getType()->isPointerTy())
    return;

  // Collect the set of return objects
  ANA.addReturnedValue(RV);
}

void MemorySummaryBuilder::visitCallInst(CallInst &I) {
  CallSite CS(&I);
  auto *Callee = CS.getCalledFunction();
  if (!Callee)
    return;

  auto *CalleeSummary = MS.getSummary(Callee);

  // Hanle black box function
  if (!CalleeSummary) {
    for (unsigned i = 0, e = CS.arg_size(); i != e; ++i) {
      auto *Ptr = CS.getArgOperand(i);

      if (!Ptr->getType()->isPointerTy() ||
          !Ptr->getType()->getPointerElementType()->isSized())
        continue;

      AST.add(Ptr, MemoryLocation::UnknownSize, AAMDNodes());
      auto Align = guessAlign(Ptr, CS.getParamAlignment(i));
      addAlignment(ReadAligns, Ptr, Align);
      addAlignment(WriteAligns, Ptr, Align);
      // FIXME: Align or size?
      PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, Align));
    }

    return;
  }

  for (auto P : CalleeSummary->interface_banks()) {
    Value *Param = const_cast<Value *>(P.first);
    auto &Bank = P.second;
    assert(Param->getType()->isPointerTy() && "Bad parameter type!");

    if (auto *Arg = dyn_cast<Argument>(Param))
      Param = CS.getArgOperand(Arg->getArgNo());

    // Add pointer to alias set
    AST.add(Param, MemoryLocation::UnknownSize, AAMDNodes());
    PtrToInsts[Param].push_back(AccessInfo::propagate(I, Param, Bank));

    // Add read or write alignment
    if (Bank.hasRefAccess())
      addAlignment(ReadAligns, Param, Bank.getMinReadAlign());

    if (Bank.hasModAccess())
      addAlignment(WriteAligns, Param, Bank.getMinWriteAlign());

    if (Bank.hasBitcast())
      BitCasts.insert(Param);
  }

  if (I.getType()->isPointerTy())
    AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());

  CurSummary.addIntrinsics(CalleeSummary->intrinsics());
}

void MemorySummaryBuilder::visitSeqBeginInst(SeqBeginInst &I) {
  unsigned TypeSizeInByte = DL.getTypeAllocSize(I.getDataType());
  uint64_t TotalSizeInBytes = MemoryLocation::UnknownSize;
  if (unsigned Length = I.getSmallConstantSize())
    TotalSizeInBytes = TypeSizeInByte * Length;

  auto *Ptr = I.getPointerOperand();
  AST.add(Ptr, TotalSizeInBytes, AAMDNodes());

  if (I.isLoad())
    addAlignment(ReadAligns, Ptr, TypeSizeInByte);
  else
    addAlignment(WriteAligns, Ptr, TypeSizeInByte);

  // Burst forced a linear address layout, we had to linearize memory as well
  BitCasts.insert(Ptr);
  PtrToInsts[Ptr].push_back(AccessInfo::seq(I, DL));
}

void MemorySummaryBuilder::visitFPGAFIFOInst(FPGAFIFOInst &I) {
  auto *FIFO = I.getFIFOOperand();
  AST.add(FIFO, MemoryLocation::UnknownSize, AAMDNodes());
  auto EltSize = DL.getTypeAllocSize(I.getDataType());
  if (I.isConsumerSide())
    addAlignment(ReadAligns, FIFO, EltSize);
  if (I.isProducerSide())
    addAlignment(WriteAligns, FIFO, EltSize);
  PtrToInsts[FIFO].push_back(AccessInfo::stream(I, DL));
}

void MemorySummaryBuilder::visitAXISIntrinsic(AXISIntrinsicInst &I) {
  for (unsigned Idx = 0; Idx < AXISIntrinsicInst::NumChannels; Idx++) {
    auto *Ch = I.getOperand(Idx);
    AST.add(Ch, MemoryLocation::UnknownSize, AAMDNodes());
    if (isa<Constant>(Ch) && cast<Constant>(Ch)->isNullValue())
      continue;

    auto EltSize = DL.getTypeAllocSize(Ch->getType()->getPointerElementType());
    if (I.isConsumerSide())
      addAlignment(ReadAligns, Ch, EltSize);
    if (I.isProducerSide())
      addAlignment(WriteAligns, Ch, EltSize);
    PtrToInsts[Ch].push_back(AccessInfo::stream(I, Idx, DL));
  }
}

void MemorySummaryBuilder::visitPipeBlockingReadInst(
    spir::PipeBlockingReadInst &I) {
  auto *Pipe = I.getPipe();
  AST.add(Pipe, MemoryLocation::UnknownSize, AAMDNodes());
  auto EltSize = DL.getTypeAllocSize(I.getDataType());
  addAlignment(ReadAligns, Pipe, EltSize);
  PtrToInsts[Pipe].push_back(AccessInfo::stream(I, DL));
}

void MemorySummaryBuilder::visitPipeBlockingWriteInst(
    spir::PipeBlockingWriteInst &I) {
  auto *Pipe = I.getPipe();
  AST.add(Pipe, MemoryLocation::UnknownSize, AAMDNodes());
  auto EltSize = DL.getTypeAllocSize(I.getDataType());
  addAlignment(WriteAligns, Pipe, EltSize);
  PtrToInsts[Pipe].push_back(AccessInfo::stream(I, DL));
}

void MemorySummaryBuilder::visitShiftRegInst(ShiftRegInst &I) {
  auto *Ptr = I.getPointerOperand();
  AST.add(Ptr, MemoryLocation::UnknownSize, AAMDNodes());

  auto *DataT = I.getDataType();
  auto EltSize = DL.getTypeAllocSize(DataT);
  addAlignment(WriteAligns, Ptr, EltSize);
  addAlignment(ReadAligns, Ptr, EltSize);
  // When it's an array of shift registger, then don't infer it as Memory
  // port(BRAM) (Because BRAM means will array-flatten and struct pack on
  // the array element type).
  ShiftRegs.insert(Ptr);
  PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, EltSize));
}

void MemorySummaryBuilder::visitMAXIStoreInst(MAXIStoreInst &I) {
  Value *Ptr = I.getPointerOperand();
  AST.add(Ptr, MemoryLocation::UnknownSize, AAMDNodes());
  auto *DataT = I.getDataType();
  auto EltSize = DL.getTypeAllocSize(DataT);
  addAlignment(WriteAligns, Ptr, EltSize);
  PtrToInsts[Ptr].push_back(AccessInfo::maxi(I, DL));
}

void MemorySummaryBuilder::visitMAXIIOInst(MAXIIOInst &I) {
  Value *Ptr = I.getPointerOperand();
  AST.add(Ptr, MemoryLocation::UnknownSize, AAMDNodes());
  auto *DataT = I.getDataType();
  auto EltSize = DL.getTypeAllocSize(DataT);
  if (I.isReadIO())
    addAlignment(ReadAligns, Ptr, EltSize);
  else
    addAlignment(WriteAligns, Ptr, EltSize);
  PtrToInsts[Ptr].push_back(AccessInfo::maxi(I, DL));
}

void MemorySummaryBuilder::visitMemTransferInst(MemTransferInst &I) {
  auto *Src = I.getRawSource(), *Dst = I.getRawDest();
  AST.add(Src, MemoryLocation::UnknownSize, AAMDNodes());
  AST.add(Dst, MemoryLocation::UnknownSize, AAMDNodes());
  auto Align = I.getAlignment();
  PtrToInsts[Src].push_back(AccessInfo::unknown(I, Align));
  PtrToInsts[Dst].push_back(AccessInfo::unknown(I, Align));

  addAlignment(WriteAligns, Dst, Align);
  // Due to the limitation in memcpy lowering, the Src pointer also need to
  // align. So we put it to write aligns to force the alignments
  addAlignment(WriteAligns, Src, Align);
  // Insert src/dst to bitcast set even if they are not, otherwise the later
  // pass cannot memcpy/memmove
  BitCasts.insert(Src);
  BitCasts.insert(Dst);
}

void MemorySummaryBuilder::visitMemSetInst(MemSetInst &I) {
  auto *Dst = I.getRawDest();
  AST.add(Dst, MemoryLocation::UnknownSize, AAMDNodes());
  auto Align = I.getAlignment();
  PtrToInsts[Dst].push_back(AccessInfo::unknown(I, Align));

  addAlignment(WriteAligns, Dst, Align);
  // Insert dst to bitcast set even if they are not, otherwise the later pass
  // cannot memset
  BitCasts.insert(Dst);
}

void MemorySummaryBuilder::visitSSACopy(SSACopyInst &I) {}

void MemorySummaryBuilder::visitIntrinsicInst(IntrinsicInst &I) {
  auto ID = I.getIntrinsicID();
  switch (ID) {
  default:
    break;
  case Intrinsic::spir_get_work_dim:
  case Intrinsic::spir_get_num_groups:
  case Intrinsic::spir_get_group_id:
  case Intrinsic::spir_get_global_offset:
  case Intrinsic::spir_get_global_id_base:
  case Intrinsic::spir_get_global_size:
  case Intrinsic::spir_get_local_id:
  case Intrinsic::spir_get_local_size:
  case Intrinsic::spir_get_local_linear_id:
  case Intrinsic::fpga_get_printf_buffer:
  case Intrinsic::spir_get_global_id:
    return Intrinsics.set(ID);
  case Intrinsic::fpga_seq_load_begin:
  case Intrinsic::fpga_seq_store_begin:
    return visitSeqBeginInst(cast<SeqBeginInst>(I));
  case Intrinsic::fpga_fifo_not_empty:
  case Intrinsic::fpga_fifo_not_full:
  case Intrinsic::fpga_fifo_pop:
  case Intrinsic::fpga_fifo_push:
  case Intrinsic::fpga_fifo_nb_pop:
  case Intrinsic::fpga_fifo_nb_push:
    return visitFPGAFIFOInst(cast<FPGAFIFOInst>(I));
  case Intrinsic::fpga_axis_valid:
  case Intrinsic::fpga_axis_ready:
  case Intrinsic::fpga_axis_pop:
  case Intrinsic::fpga_axis_nb_pop:
  case Intrinsic::fpga_axis_push:
  case Intrinsic::fpga_axis_nb_push:
    return visitAXISIntrinsic(cast<AXISIntrinsicInst>(I));
  case Intrinsic::spir_read_pipe_block_2:
    return visitPipeBlockingReadInst(cast<spir::PipeBlockingReadInst>(I));
  case Intrinsic::spir_write_pipe_block_2:
    return visitPipeBlockingWriteInst(cast<spir::PipeBlockingWriteInst>(I));
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
    return visitMemTransferInst(cast<MemTransferInst>(I));
  case Intrinsic::memset:
    return visitMemSetInst(cast<MemSetInst>(I));
  case Intrinsic::ssa_copy:
    return visitSSACopy(cast<SSACopyInst>(I));
  case Intrinsic::fpga_shift_register_peek:
  case Intrinsic::fpga_shift_register_shift:
    return visitShiftRegInst(cast<ShiftRegInst>(I));
  case Intrinsic::fpga_pppo_load:
    return visitPPPOLoadInst(cast<FPGAPPPOLoadInst>(I));
  case Intrinsic::fpga_pppo_store:
    return visitPPPOStoreInst(cast<FPGAPPPOStoreInst>(I));
  case Intrinsic::fpga_maxi_store:
   return visitMAXIStoreInst(cast<MAXIStoreInst>(I));
  case Intrinsic::fpga_maxi_read_req:
  case Intrinsic::fpga_maxi_read:
  case Intrinsic::fpga_maxi_write_req:
  case Intrinsic::fpga_maxi_write:
  case Intrinsic::fpga_maxi_write_resp:
    return visitMAXIIOInst(cast<MAXIIOInst>(I));
  case Intrinsic::sideeffect:
    return visitSideeffect(I);
  }
}

void MemorySummaryBuilder::visitSideeffect(IntrinsicInst &I) {
  if (isa<AggregateInst>(&I))
    return visitAggregateInst(*cast<AggregateInst>(&I));
}

void MemorySummaryBuilder::visitAggregateInst(AggregateInst &I) {
  auto *V = I.getVariable();
  if (V->getType()->isPointerTy())
    BitCasts.insert(V);
}

void MemorySummaryBuilder::visitLoadInst(LoadInst &I) {
  AST.add(&I);
  auto *Ptr = I.getPointerOperand();

  auto Align = guessAlign(Ptr, I.getAlignment());
  auto *T = I.getType();
  auto EltSize = DL.getTypeStoreSize(T);
  Align = GreatestCommonDivisor64(Align, DL.getPrefTypeAlignment(T));
  Align = GreatestCommonDivisor64(Align, EltSize);
  addAlignment(ReadAligns, Ptr, Align);
  PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, EltSize));

  // Also add the result of the load if it is a pointer
  if (I.getType()->isPointerTy()) {
    AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
    BitCasts.insert(&I);
  }
}

void MemorySummaryBuilder::visitStoreInst(StoreInst &I) {
  AST.add(&I);
  auto *Ptr = I.getPointerOperand();

  auto Align = guessAlign(Ptr, I.getAlignment());
  // Also consider the size of the type to avoid writing too much
  auto *T = I.getValueOperand()->getType();
  Align = GreatestCommonDivisor64(Align, DL.getPrefTypeAlignment(T));
  auto EltSize = DL.getTypeStoreSize(T);
  Align = GreatestCommonDivisor64(Align, EltSize);
  addAlignment(WriteAligns, Ptr, Align);
  PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, EltSize));
}

void MemorySummaryBuilder::visitPPPOLoadInst(FPGAPPPOLoadInst &I) {
  AST.add(&I);
  auto Ptr = I.getPointerOperand();

  auto Align = guessAlign(Ptr, I.getAlignment());
  auto DataT = I.getDataType();
  auto EltSize = DL.getTypeStoreSize(DataT);
  Align = GreatestCommonDivisor64(Align, EltSize);
  addAlignment(ReadAligns, Ptr, Align);
  PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, EltSize));
}

void MemorySummaryBuilder::visitPPPOStoreInst(FPGAPPPOStoreInst &I) {
  AST.add(&I);
  auto Ptr = I.getPointerOperand();

  auto Align = guessAlign(Ptr, I.getAlignment());
  auto DataT = I.getDataType();
  auto EltSize = DL.getTypeStoreSize(DataT);
  Align = GreatestCommonDivisor64(Align, EltSize);
  addAlignment(WriteAligns, Ptr, Align);
  PtrToInsts[Ptr].push_back(AccessInfo::unknown(I, EltSize));
}

void MemorySummaryBuilder::visitAllocaInst(AllocaInst &I) {
  AST.add(&I, MemoryLocation::UnknownSize, AAMDNodes());
}

static const char *TranslateAccessLattice(MemorySummary::AccessLattice L) {
  switch (L) {
  case MemoryBank::NoAccess:
    return "None";
  case MemoryBank::RefAccess:
    return "Ref";
  case MemoryBank::ModAccess:
    return "Mod";
  case MemoryBank::ModRefAccess:
    return "ModRef";
  }

  llvm_unreachable("broken type!");
  return "<bad type>";
}

void MemorySummaryWrapperPass::print(raw_ostream &OS, const Module *M) const {
  if (!MS)
    return;

  if (!M) {
    MS->print(OS);
    return;
  }

  for (auto &F : *M)
    if (auto *S = MS->getSummary(&F))
      S->print(&F, OS);

  OS << "GlobalVariable ModRefs:\n";
  for (auto &G : M->globals())
    OS.indent(2) << G.getName() << " ["
                 << TranslateAccessLattice(MS->getGlobalAccess(&G)) << "]\n";
}

void MemorySummary::print(raw_ostream &OS) const {
  for (const auto &P : Summaries)
    P.second.print(P.first, OS);

  OS << "GlobalVariable ModRefs:\n";
  for (const auto P : GlobalAccesses)
    OS.indent(2) << P.first->getName() << " ["
                 << TranslateAccessLattice(AccessLattice(P.second)) << "]\n";
}

LLVM_DUMP_METHOD void MemorySummary::dump() const { print(dbgs()); }

void FunctionMemorySummary::print(const Function *F, raw_ostream &OS) const {
  StringRef Name = "<nullptr>";
  if (F)
    Name = F->getName();

  OS << "Memorybanks for Function: " << Name << '\n';
  for (auto &Pair : llvm::enumerate(Banks)) {
    if (Pair.index() + 1 == ReturnIdx)
      OS << "[Return] ";
    Pair.value().print(OS);
  }

  OS << "Intrinsics for Function: " << Name << '\n';
  for (auto I : Intrinsics)
    OS.indent(2) << Intrinsic::lookupName(Intrinsic::ID(I)) << '\n';
  OS << "Array Geomerties for Function: " << Name << '\n';
  for (auto P : ArrayGeometries)
    OS.indent(2) << *P.first << ": " << *P.second << '\n';
  OS << '\n';
}

LLVM_DUMP_METHOD void FunctionMemorySummary::dump() const {
  print(nullptr, dbgs());
}

void MemorySummaryWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<CallGraphWrapperPass>();
  getafgpaAAResultsAnalysisUsage(AU);
  AU.setPreservesAll();
}

AAResults MemorySummaryWrapperPass::createAAResults(Function &F,
                                                    ArgumentNoAlias &ANA) {
  AAResults AAR(getAnalysis<TargetLibraryInfoWrapperPass>().getTLI());

  AAR.addAAResult(ANA);

  // BasicAA is always available for function analyses. Also, we add it first
  // so that it can trump TBAA results when it proves MustAlias.
  // FIXME: TBAA should have an explicit mode to support this and then we
  // should reconsider the ordering here.
  // if (!DisableBasicAA)
  AAR.addAAResult(getAnalysis<BasicAAWrapperPass>(F).getResult());

  // Populate the results with the currently available AAs.
  if (auto *WrapperPass = getAnalysisIfAvailable<ScopedNoAliasAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<TypeBasedAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  // if (auto *WrapperPass =
  //     getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
  //   AAR->addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<GlobalsAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  // if (auto *WrapperPass = getAnalysisIfAvailable<SCEVAAWrapperPass>())
  //   AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<CFLAndersAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());
  if (auto *WrapperPass = getAnalysisIfAvailable<CFLSteensAAWrapperPass>())
    AAR.addAAResult(WrapperPass->getResult());

  return AAR;
}

void MemorySummaryWrapperPass::runOnFunction(Function &F, const DataLayout &DL,
                                             GlobalVariableEC &GEC) {
  DEBUG(dbgs() << "Run on function: " << F.getName() << '\n');
  ArgumentNoAlias ANA(DL, GEC, *MS);

  auto AA = createAAResults(F, ANA);
  MemorySummaryBuilder MSB(*MS, F, AA, ANA);
  MSB.run();
  DEBUG(MSB.dump());

  // Build the alias set of global variables
  for (auto &AS : MSB.AST) {
    if (AS.isForwardingAliasSet())
      continue;

    GlobalVariable *Leader = nullptr;
    for (auto V : AS) {
      auto *GV = dyn_cast<GlobalVariable>(V.getValue());
      if (!GV)
        continue;

      if (!Leader) {
        Leader = GV;
        continue;
      }

      GEC.unionSets(Leader, GV);
    }
  }
}

void MemorySummaryWrapperPass::runOnCallGraphPostOrder(
    CallGraphNode *Root, const DataLayout &DL,
    std::set<CallGraphNode *> &Visited, GlobalVariableEC &GEC) {
  for (auto *N : post_order_ext(Root, Visited)) {
    auto *F = N->getFunction();
    if (!F || F->isDeclaration())
      continue;

    runOnFunction(*F, DL, GEC);
  }
}

static void InitializeGEC(Module &M, GlobalVariableEC &GEC,
                          const DataLayout &DL) {
  for (auto &G : M.globals())
    GEC.insert(&G);

  // Hanle the address look-up table as well
  SmallVector<GlobalVariable *, 8> Objs;
  for (auto &G : M.globals()) {
    if (!ExtractUnderlyingObject(G, DL, Objs))
      continue;

    auto *Leader = Objs.pop_back_val();
    while (!Objs.empty())
      GEC.unionSets(Leader, Objs.pop_back_val());
  }
}

bool MemorySummaryWrapperPass::runOnModule(Module &M) {
  MS.reset(new MemorySummary());
  CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
  const auto &DL = M.getDataLayout();

  GlobalVariableEC GEC;
  InitializeGEC(M, GEC, DL);

  // getExternalCallingNode
  std::set<CallGraphNode *> Visited;
  runOnCallGraphPostOrder(CG.getExternalCallingNode(), DL, Visited, GEC);

  // Cover the function that are not reachable from the ExternalCallingNode,
  // i.e. the dead functions. This pass should not depends any optimization
  // So we just handle them
  for (auto &P : CG)
    runOnCallGraphPostOrder(&*P.second, DL, Visited, GEC);

  return false;
}

MemorySummaryWrapperPass::MemorySummaryWrapperPass(bool RM)
    : ModulePass(ID), ShouldReleaseMemory(RM) {
  initializeMemorySummaryWrapperPassPass(*PassRegistry::getPassRegistry());
}

char MemorySummaryWrapperPass::ID = 0;
INITIALIZE_PASS_BEGIN(MemorySummaryWrapperPass, DEBUG_TYPE, "Memory Summary",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(MemorySummaryWrapperPass, DEBUG_TYPE, "Memory Summary",
                    false, false)

Pass *llvm::createMemorySummaryWrapperPassPass() {
  return new MemorySummaryWrapperPass();
}

void FunctionMemorySummary::addMemoryBank(MemoryBank &&MB, bool IsRet) {
  unsigned Idx = Banks.size() + 1;

  for (Value *V : make_filter_range(MB.objects(), [](Value *V) {
         return isa<Argument>(V) || isa<GlobalVariable>(V);
       }))
    BankIdx.insert({V, Idx});

  if (IsRet)
    ReturnIdx = Idx;

  Banks.emplace_back(MB);
}

void FunctionMemorySummary::addArrayGeometry(const Argument *A, ArrayType *T) {
  bool Insert = ArrayGeometries.insert({A, T}).second;
  assert(Insert && "Cannot add array geometries!");
  (void)Insert;
}

void FunctionMemorySummary::addIntrinsic(Intrinsic::ID ID) {
  Intrinsics.set(ID);
}

void FunctionMemorySummary::addIntrinsics(const IntrinsicSet &S) {
  Intrinsics |= S;
}

void FunctionMemorySummary::IterateIntrinsicXYZ(
    StringRef Name, unsigned SizeInBits,
    function_ref<void(StringRef, unsigned)> Fn) {
  SmallString<8> S = Name;

  S += "_x";
  Fn(S, SizeInBits);
  S.resize(Name.size());

  S += "_y";
  Fn(S, SizeInBits);
  S.resize(Name.size());

  S += "_z";
  Fn(S, SizeInBits);
}

void FunctionMemorySummary::IterateIntrinsics(
    const DataLayout &DL, IntrinsicSet Set,
    function_ref<void(StringRef, unsigned)> Fn) {
  if (Set.test(Intrinsic::spir_get_work_dim))
    Fn("work_dim", 32);
  if (Set.test(Intrinsic::spir_get_global_size))
    IterateIntrinsicXYZ("global_size", 32, Fn);
  if (Set.test(Intrinsic::spir_get_local_size))
    IterateIntrinsicXYZ("local_size", 32, Fn);
  if (Set.test(Intrinsic::spir_get_num_groups))
    IterateIntrinsicXYZ("num_groups", 32, Fn);
  if (Set.test(Intrinsic::spir_get_group_id))
    IterateIntrinsicXYZ("group_id", 32, Fn);
  if (Set.test(Intrinsic::spir_get_global_offset))
    IterateIntrinsicXYZ("global_offset", 32, Fn);
  if (Set.test(Intrinsic::fpga_get_printf_buffer))
    Fn("printf_buffer", DL.getPointerSizeInBits());
  if (Set.test(Intrinsic::spir_get_global_id_base))
    IterateIntrinsicXYZ("global_id_base", 32, Fn);
}

void FunctionMemorySummary::iterateIntrinsics(
    const DataLayout &DL, function_ref<void(StringRef, unsigned)> Fn) const {
  IterateIntrinsics(DL, intrinsics(), Fn);
}

const Value *
FunctionMemorySummary::getReturnBankRepresentative(ImmutableCallSite I) const {
  auto *Bank = getReturnBank();

  if (!Bank)
    return nullptr;
  Value *O = Bank->objects().front();
  if (isa<Constant>(O))
    return O;

  // Need to translate formal parameters to actual parameters
  auto *A = cast<Argument>(O);
  return I.getArgOperand(A->getArgNo());
}

AnalysisKey MemorySummaryAnalysis::Key;

AAResults MemorySummaryAnalysis::createAAResults(Function &F,
                                                 ModuleAnalysisManager &MAM,
                                                 ArgumentNoAlias &ANA) {
  auto &M = *F.getParent();
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AAResults AAR(MAM.getResult<TargetLibraryAnalysis>(M));

  AAR.addAAResult(ANA);

  // BasicAA is always available for function analyses. Also, we add it first
  // so that it can trump TBAA results when it proves MustAlias.
  // FIXME: TBAA should have an explicit mode to support this and then we
  // should reconsider the ordering here.
  // if (!DisableBasicAA)
  AAR.addAAResult(FAM.getResult<BasicAA>(F));

  // Populate the results with the currently available AAs.
  if (auto *Result = FAM.getCachedResult<ScopedNoAliasAA>(F))
    AAR.addAAResult(*Result);
  if (auto *Result = FAM.getCachedResult<TypeBasedAA>(F))
    AAR.addAAResult(*Result);
  // if (auto *WrapperPass =
  //     getAnalysisIfAvailable<objcarc::ObjCARCAAWrapperPass>())
  //   AAR->addAAResult(Result);
  if (auto *Result = MAM.getCachedResult<GlobalsAA>(M))
    AAR.addAAResult(*Result);
  // if (auto *WrapperPass = getAnalysisIfAvailable<SCEVAAWrapperPass>())
  //   AAR.addAAResult(Result);
  if (auto *Result = FAM.getCachedResult<CFLAndersAA>(F))
    AAR.addAAResult(*Result);
  if (auto *Result = FAM.getCachedResult<CFLSteensAA>(F))
    AAR.addAAResult(*Result);
  return AAR;
}

void MemorySummaryAnalysis::runOnFunction(Function &F, const DataLayout &DL,
                                          ModuleAnalysisManager &MAM,
                                          GlobalVariableEC &GEC,
                                          MemorySummary &MS) {
  DEBUG(dbgs() << "Run on function: " << F.getName() << '\n');

  ArgumentNoAlias ANA(DL, GEC, MS);

  auto AA = createAAResults(F, MAM, ANA);
  MemorySummaryBuilder MSB(MS, F, AA, ANA);
  MSB.run();
  DEBUG(MSB.dump());

  // Build the alias set of global variables
  for (auto &AS : MSB.AST) {
    if (AS.isForwardingAliasSet())
      continue;

    GlobalVariable *Leader = nullptr;
    for (auto V : AS) {
      auto *GV = dyn_cast<GlobalVariable>(V.getValue());
      if (!GV)
        continue;

      if (!Leader) {
        Leader = GV;
        continue;
      }

      GEC.unionSets(Leader, GV);
    }
  }
}

void MemorySummaryAnalysis::runOnCallGraphPostOrder(
    CallGraphNode *Root, const DataLayout &DL, ModuleAnalysisManager &MAM,
    GlobalVariableEC &GEC, MemorySummary &MS,
    std::set<CallGraphNode *> &Visited) {
  for (auto *N : post_order_ext(Root, Visited)) {
    auto *F = N->getFunction();
    if (!F || F->isDeclaration())
      continue;

    runOnFunction(*F, DL, MAM, GEC, MS);
  }
}

MemorySummary MemorySummaryAnalysis::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  MemorySummary MS;
  auto &CG = MAM.getResult<CallGraphAnalysis>(M);
  auto &DL = M.getDataLayout();

  GlobalVariableEC GEC;
  for (auto &G : M.globals())
    GEC.insert(&G);

  // getExternalCallingNode
  std::set<CallGraphNode *> Visited;
  runOnCallGraphPostOrder(CG.getExternalCallingNode(), DL, MAM, GEC, MS,
                          Visited);

  // Cover the function that are not reachable from the ExternalCallingNode,
  // i.e. the dead functions. This pass should not depends any optimization
  // So we just handle them
  for (auto &P : CG)
    runOnCallGraphPostOrder(&*P.second, DL, MAM, GEC, MS, Visited);

  return MS;
}

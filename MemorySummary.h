//===-- reflow/Memory/MemorySummary.h - Summarize the Memory Banking - C++ -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declare the analysis to identify the memory banks in the current
// LLVM module
//
//===----------------------------------------------------------------------===//

#ifndef REFLOW_MEMORY_MEMORY_SUMMARY_H
#define REFLOW_MEMORY_MEMORY_SUMMARY_H

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/XILINXAggregateUtil.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"


#include <set>

namespace llvm {
class AliasSet;

class MemoryBank {
public:
  /// The kinds of access this alias set models.
  ///
  /// We keep track of whether this alias set merely refers to the locations of
  /// memory (and not any particular access), whether it modifies or references
  /// the memory, or whether it does both. The lattice goes from "NoAccess" to
  /// either RefAccess or ModAccess, then to ModRefAccess as necessary.
  enum AccessLattice {
    NoAccess = 0,
    RefAccess = 1,
    ModAccess = 2,
    ModRefAccess = RefAccess | ModAccess
  };

  enum : uint64_t { UnknownSize = MemoryLocation::UnknownSize };

private:
  SmallVector<AssertingVH<Value>, 8> Objs;
  SmallVector<TrackingVH<Instruction>, 8> Insts;
  uint64_t MinReadAlign, MinWriteAlign;

  // Maximum alignment on all the underlying objects and all accesses.
  // Initialize with underlying object elemet alloc size, and LCM all accesses.
  uint64_t MaxAlign;

  // Maximum accesses size. max of all accesses.
  uint64_t MaxSize;

  // Minimum accesses size. min of all accesses.
  uint64_t MinSize;

  // Minimum type abi alignment of the underlying object list.
  uint64_t MinTypeAlign;

  // Maximum type alloc size of the underlying object list.
  uint64_t MaxTypeSize;

  // exact type size in bits when type is integer type or the struct type
  // recursively has only one filed, otherwise, the value is the same with \v
  // MaxTypeSize * 8
  uint64_t PreciseTypeSizeInBits;

  // Minimum type store size. GCD store size of all writes.
  uint64_t MinTypeSizeFromStore;

  unsigned AS;
  bool HasBitCast;
  bool IsStream;
  bool IsMAXI;
  bool IsShiftReg;
  bool IsOnlyOneField;
  uint64_t SeqReadSizeInBytes, SeqWriteSizeInBytes;
  unsigned /*AccessLattice*/ ModRef;
  AggregateInfo AggrInfo;

  void addAccessSize(uint64_t Size) {
    MaxSize = std::max(MaxSize, Size);
    MinSize = GreatestCommonDivisor64(MinSize, Size);
  }

  void updateMinTypeSizeFromStore(Value *Ptr, const DataLayout &DL);

  void setAccessMode(unsigned Access) { ModRef = Access; }

  void addReadAlign(uint64_t Algin);

  void addWriteAlign(uint64_t Algin);

  void addInsts(ArrayRef<Instruction*> A) {
    Insts.append(A.begin(), A.end());
  }

  void initMaxAlign(const DataLayout &DL);
  void initMinTypeAlign(const DataLayout &DL);
  void initMaxTypeSize(const DataLayout &DL);
  void initPreciseTypeSize(const DataLayout &DL);
  void initOnlyOneField();

  friend struct MemorySummaryBuilder;

public:
  static bool IsObject(Value *V);

  MemoryBank(const AliasSet &AS, bool HasBitCast, bool IsShiftReg);

  ArrayRef<AssertingVH<Value>> objects() const { return Objs; }
  ArrayRef<TrackingVH<Instruction>> insts() const { return Insts; }
  Value *getFirstObject() const { return Objs.front(); }
  Value *getUniqueObject() const {
    if (Objs.size() != 1)
      return nullptr;
    return Objs.front();
  }
  size_t objects_size() const { return Objs.size(); }

  uint64_t getMinWriteAlign() const { return MinWriteAlign; }
  uint64_t getMinReadAlign() const { return MinReadAlign; }
  uint64_t getMaxAlign() const { return MaxAlign; }
  uint64_t getMaxSize() const { return MaxSize; }
  uint64_t getMinSize() const { return MinSize; }
  uint64_t getMinTypeAlign() const { return MinTypeAlign; }
  uint64_t getMaxTypeSize() const { return MaxTypeSize; }
  uint64_t getPreciseTypeSizeInBits() const { return PreciseTypeSizeInBits; }
  uint64_t getMinTypeSizeFromStore() const { return MinTypeSizeFromStore; }

  unsigned getAddressSpace() const { return AS; }

  AccessLattice getAccessMode() const { return AccessLattice(ModRef); }
  bool hasRefAccess() const {
    return (getAccessMode() & RefAccess) != NoAccess;
  }
  bool hasModAccess() const {
    return (getAccessMode() & ModAccess) != NoAccess;
  }

  bool hasBitcast() const { return HasBitCast; }
  bool hasOnlyOneField() const { return IsOnlyOneField; }
  bool isStream() const { return IsStream; }
  bool isMAXI() const { return IsMAXI; }
  bool isShiftReg() const { return IsShiftReg; }
  uint64_t getSeqReadSizeInBytes() const { return SeqReadSizeInBytes; }
  uint64_t getSeqWriteSizeInBytes() const { return SeqWriteSizeInBytes; }

  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

class CallGraph;
class CallGraphNode;
class ReflowAAResult;
class ArgumentNoAlias;
class AAResults;
class MemorySummaryWrapperPass;

class FunctionMemorySummary {
public:
  using IntrinsicSet = SparseBitVector<>;

private:
  using InterfaceBankIdxMapTy = MapVector<AssertingVH<const Value>, unsigned>;
  using BankIdxPairTy = std::pair<AssertingVH<const Value>, unsigned>;
  using DerefFun =
      std::pointer_to_unary_function<const BankIdxPairTy &, const Value *>;
  static const Value *InterfaceValueDereference(const BankIdxPairTy &P) {
    return P.first;
  }

  using ValueBankPair = std::pair<const Value *, const MemoryBank &>;
  using MapFun = std::function<ValueBankPair(const BankIdxPairTy &)>;

  ValueBankPair InterfaceBankMapping(const BankIdxPairTy &P) const {
    return ValueBankPair(P.first, Banks[P.second - 1]);
  }

  SmallVector<MemoryBank, 8> Banks;
  InterfaceBankIdxMapTy BankIdx;
  unsigned ReturnIdx = 0;
  IntrinsicSet Intrinsics;
  DenseMap<AssertingVH<const Argument>, ArrayType *> ArrayGeometries;

  void addMemoryBank(MemoryBank &&MB, bool IsRet);
  void addArrayGeometry(const Argument *A, ArrayType *T);

  void addIntrinsic(Intrinsic::ID ID);
  void addIntrinsics(const IntrinsicSet &S);

  static void IterateIntrinsicXYZ(StringRef Name, unsigned SizeInBits,
                                  function_ref<void(StringRef, unsigned)> Fn);

  friend struct MemorySummaryBuilder;
  friend class MemorySummaryWrapperPass;

public:
  ArrayRef<MemoryBank> banks() const { return Banks; }

  IntrinsicSet intrinsics() const { return Intrinsics; }

  bool inInterfaceBank(const Value *V) const { return BankIdx.count(V); }

  const MemoryBank &getInterfaceBank(const Value *V) const {
    unsigned Idx = BankIdx.lookup(V);
    assert(Idx && "Cannot find interface bank!");
    return Banks[Idx - 1];
  }

  const MemoryBank *getReturnBank() const {
    if (ReturnIdx)
      return &Banks[ReturnIdx - 1];

    return nullptr;
  }

  const Value *getReturnBankRepresentative(ImmutableCallSite I) const;

  bool interface_empty() const { return Banks.empty(); }

  using const_interface_value_iterator =
      mapped_iterator<InterfaceBankIdxMapTy::const_iterator, DerefFun>;
  const_interface_value_iterator interface_value_begin() const {
    return const_interface_value_iterator(BankIdx.begin(),
                                          DerefFun(InterfaceValueDereference));
  }
  const_interface_value_iterator interface_value_end() const {
    return const_interface_value_iterator(BankIdx.end(),
                                          DerefFun(InterfaceValueDereference));
  }
  iterator_range<const_interface_value_iterator> interface_values() const {
    return make_range(interface_value_begin(), interface_value_end());
  }

  using const_interface_bank_iterator =
      mapped_iterator<InterfaceBankIdxMapTy::const_iterator, MapFun>;
  const_interface_bank_iterator interface_bank_begin() const {
    return const_interface_bank_iterator(
        BankIdx.begin(),
        MapFun(std::bind(&FunctionMemorySummary::InterfaceBankMapping, this,
                         std::placeholders::_1)));
  }
  const_interface_bank_iterator interface_bank_end() const {
    return const_interface_bank_iterator(
        BankIdx.end(),
        MapFun(std::bind(&FunctionMemorySummary::InterfaceBankMapping, this,
                         std::placeholders::_1)));
  }
  iterator_range<const_interface_bank_iterator> interface_banks() const {
    return make_range(interface_bank_begin(), interface_bank_end());
  }

  ArrayType *lookupGeometry(const Argument *A) const {
    return ArrayGeometries.lookup(A);
  }

  static void IterateIntrinsics(const DataLayout &DL, IntrinsicSet Set,
                                function_ref<void(StringRef, unsigned)> Fn);

  void iterateIntrinsics(const DataLayout &DL,
                         function_ref<void(StringRef, unsigned)> Fn) const;

  void print(const Function *F, raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

class MemorySummary {
  DenseMap<AssertingVH<const Function>, FunctionMemorySummary> Summaries;
  DenseMap<AssertingVH<const GlobalVariable>, unsigned> GlobalAccesses;

  void addSummary(const Function *F, FunctionMemorySummary &&FS) {
    bool Inserted = Summaries.insert({F, FS}).second;
    assert(Inserted && "Cannot insert!");
    (void)Inserted;
  }

  void addGlobalAccess(const GlobalVariable *GV, unsigned Access) {
    GlobalAccesses[GV] |= Access;
  }

  friend struct MemorySummaryBuilder;
  friend class MemorySummaryWrapperPass;

public:
  using AccessLattice = MemoryBank::AccessLattice;

  using GlobalVariableEC = EquivalenceClasses<const GlobalVariable *>;

  const FunctionMemorySummary *getSummary(const Function *F) const {
    auto I = Summaries.find(F);
    return (I != Summaries.end()) ? &I->second : nullptr;
  }

  AccessLattice getGlobalAccess(const GlobalVariable *GV) const {
    return AccessLattice(GlobalAccesses.lookup(GV) &
                         AccessLattice::ModRefAccess);
  }

  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

//===----------------------------------------------------------------------===//
/// \brief The legacy pass manager's analysis pass for MemorySummary
class MemorySummaryWrapperPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  using GlobalVariableEC = MemorySummary::GlobalVariableEC;

  MemorySummaryWrapperPass(bool RM = true);

  ~MemorySummaryWrapperPass() { releaseMemory(); }

  void runOnFunction(Function &F, const DataLayout &DL, GlobalVariableEC &GEC);
  void runOnCallGraphPostOrder(CallGraphNode *Root, const DataLayout &DL,
                               std::set<CallGraphNode *> &Visited,
                               GlobalVariableEC &GEC);

  /// \brief Calculate all the polyhedral scops for a given function.
  bool runOnModule(Module &M) override;

  void print(raw_ostream &O, const Module *M = nullptr) const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  const MemorySummary &getMemorySummary() const { return *MS; }

  void releaseMemory() override {
    if (!ShouldReleaseMemory)
      return;
    MS.reset();
  }

private:
  bool ShouldReleaseMemory = true;
  AAResults createAAResults(Function &F, ArgumentNoAlias &ANA);
  std::unique_ptr<MemorySummary> MS;
};

/// \brief The wrapper of MemorySummary for LLVM Pass Pipeline
class MemorySummaryAnalysis : public AnalysisInfoMixin<MemorySummaryAnalysis> {
  using GlobalVariableEC = MemorySummary::GlobalVariableEC;

  AAResults createAAResults(Function &F, ModuleAnalysisManager &MAM,
                            ArgumentNoAlias &ANA);
  void runOnFunction(Function &F, const DataLayout &DL,
                     ModuleAnalysisManager &MAM, GlobalVariableEC &GEC,
                     MemorySummary &MS);
  void runOnCallGraphPostOrder(CallGraphNode *Root, const DataLayout &DL,
                               ModuleAnalysisManager &MAM,
                               GlobalVariableEC &GEC, MemorySummary &MS,
                               std::set<CallGraphNode *> &Visited);

public:
  static AnalysisKey Key;

  using Result = MemorySummary;

  Result run(Module &, ModuleAnalysisManager &);
};

Pass *createMemorySummaryWrapperPassPass();
void initializeMemorySummaryWrapperPassPass(PassRegistry &);
} // namespace llvm

#endif // !OWL_ANALYSIS_MEMORY_SUMMARY_H

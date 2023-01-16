//===----------------------------------------------------------------------===//
//
// This file declares common functions useful for getting use-def/def-use
// information in XILINX HLS IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_XILINXHLSVALUETRACKINGUTILS_H
#define LLVM_ANALYSIS_XILINXHLSVALUETRACKINGUTILS_H

#include "llvm/ADT/SmallVector.h"

namespace llvm {

class Value;
class DataLayout;
class LoopInfo;
class GlobalValue;
class Type;

bool IsGlobalUseEmpty(const GlobalValue &GV);

bool IsHLSStream(const Value *V);

Type *StripPadding(Type *T, const DataLayout &DL);

Type *extractHLSStreamEltType(Type *T);

/// Same as llvm::GetUnderlyingObject but stops on ssa_copy
Value *GetUnderlyingSSACopyOrUnderlyingObject(Value *V, const DataLayout &DL,
                                              unsigned MaxLookup = 6);

/// Same as llvm::GetUnderlyingObjects but stops on ssa_copy
void GetUnderlyingSSACopiesOrUnderlyingObjects(
    Value *V, SmallVectorImpl<Value *> &Objects, const DataLayout &DL,
    LoopInfo *LI = nullptr, unsigned MaxLookup = 6);

/// Returns an ssa_copy or the final underlying object or nullptr if ambiguous
Value *GetUniqueSSACopyOrUnderlyingObject(Value *V, const DataLayout &DL,
                                          LoopInfo *LI, unsigned MaxLookup = 6);

/// Returns a chain of ssa_copy finished by the final underlying object, or
/// finished by nullptr if ambiguous at that level
SmallVector<Value *, 4> CollectSSACopyChain(Value *V, const DataLayout &DL,
                                            LoopInfo *LI,
                                            unsigned MaxLookup = 6);

} // end namespace llvm

#endif // LLVM_ANALYSIS_XILINXHLSVALUETRACKINGUTILS_H

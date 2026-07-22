class AtomicStoreOpLowering : public OpRewritePattern<hivm::StoreOp> {
  using OpRewritePattern<hivm::StoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::StoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = getElementTypeOrSelf(op.getDstOperandType());
    auto atomicKind = op.getAtomicKind();
    if (op.isAtomic()) {
      if (elemType.isInteger(64))
        return decomposeEltwiseAtomic(op, rewriter, loc);
      if (*atomicKind == hivm::AtomicKind::UMAX ||
          *atomicKind == hivm::AtomicKind::UMIN) {
        assert(elemType.getIntOrFloatBitWidth() == 8);
        return decomposeEltwiseAtomic(op, rewriter, loc, /*isUnsigned=*/true);
      }
      if ((*atomicKind == hivm::AtomicKind::ADD ||
           *atomicKind == hivm::AtomicKind::MAX ||
           *atomicKind == hivm::AtomicKind::MIN) &&
          isAtomicOpHaveReturnedValue(op)) {
        return addSyncForReturnedValue(op, rewriter, loc);
      }
    }
    if (!op.isSWAtomic()) {
      return failure();
    }
    switch (atomicKind.value()) {
    case hivm::AtomicKind::AND:
    case hivm::AtomicKind::OR:
    case hivm::AtomicKind::XOR:
      return decomposeEltwiseAtomic(op, rewriter, loc);
    default:
      return failure();
    }
  }

private:
  /// Find the load operation used to save the returned value before atomic operation being calculated
  /// e.g. hivm.hir.load ins(%reinterpret_cast) outs(%alloc)
  /// hivm.hir.store ins(%cast) outs(%reinterpret_cast)
  ///
  /// The load operation whose outs operand is same as store's ins operand is required
  Operation* findReturnedValueLoadOp(hivm::StoreOp storeOp, Value targetValue) const {
    if (!targetValue) return nullptr;

    Operation* op = storeOp->getPrevNode();
    while (op) {
      if (auto loadOp = dyn_cast<hivm::LoadOp>(op)) {
        if (loadOp->getOperand(0) == targetValue) {
          return loadOp;
        }
      }
      op = op->getPrevNode();
    }

    return nullptr;
  }

  bool isAtomicOpHaveReturnedValue(hivm::StoreOp storeOp) const {
    Operation* returnedValueLoadOp = findReturnedValueLoadOp(storeOp, storeOp.getDst());
    if (auto LoadOp = dyn_cast_or_null<hivm::LoadOp>(returnedValueLoadOp)) {
      auto dst = LoadOp.getDst();
      // If the dst of loadOp just be used for this loadop, the returned value dead code.
      return llvm::range_size(dst.getUsers()) > 1;
    }
    return false;
  }

  LogicalResult addSyncForReturnedValue(hivm::StoreOp op,
                                        PatternRewriter &rewriter, Location loc) const {
    static constexpr llvm::StringLiteral kAlreadySync =
        "already_sync";
    if (op->hasAttr(kAlreadySync)) {
      return failure();
    }
    Operation* returnedValueLoadOp = findReturnedValueLoadOp(op, op.getDst());
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(returnedValueLoadOp);

    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    ::mlir::Value val = lockVar;
    Operation *lockDefOp = val.getDefiningOp();
    auto createLockOp = dyn_cast<hivm::CreateSyncBlockLockOp>(lockDefOp);
    if (createLockOp) {
        createLockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());
    }
    auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    lockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    rewriter.setInsertionPointAfter(op);
    auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    unlockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    op->setAttr(kAlreadySync, UnitAttr::get(op->getContext()));
    return success();
  }

  /// implement atomic by software way
  /// e.g.store ins(% res_ub) outs(% res_gm) with atomic XOR is converted to
  /// % lock_var = create_sync_lock()
  /// sync_block_lock(% lock_var)
  ///
  /// % tmp0_ub = load % res_gm % tmp0_ub =
  /// % tmp0_ub xor % res_ub
  /// store ins(% tmp0_ub) outs(% res_gm)
  ///
  /// sync_block_unlock(% lock_var)
  LogicalResult decomposeEltwiseAtomic(hivm::StoreOp op,
                                       PatternRewriter &rewriter, Location loc,
                                       bool isUnsigned = false) const {
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    ::mlir::Value val = lockVar;
    Operation *lockDefOp = val.getDefiningOp();
    auto createLockOp = dyn_cast<hivm::CreateSyncBlockLockOp>(lockDefOp);
    if (createLockOp) {
        createLockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());
    }
    auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    lockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    // 2. create tmp memref alloc and load dst to tmp
    auto src = op.getSrc();
    auto tmpUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    if (isUnsigned) {
      hivm::RoundMode rounding = mlir::utils::selectRoundMode<hivm::RoundMode>(
          getElementTypeOrSelf(dst), rewriter.getF16Type());
      auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
      hivm::TypeFn typeFn = hivm::TypeFn::cast_unsigned;
      auto typeFnAttr = rewriter.getAttr<hivm::TypeFnAttr>(typeFn);
      src = castTo(rewriter, src.getLoc(), src, roundingAttr,
                   rewriter.getF16Type(), typeFnAttr)
                .getSingleDst();
      tmpUB = castTo(rewriter, src.getLoc(), tmpUB, roundingAttr,
                     rewriter.getF16Type(), typeFnAttr)
                  .getSingleDst();
    }

    // 3. do eltwise vv between src and tmp(and/or/xor)
    auto resUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);
    auto eltwiseOp = createEltwiseOpByAtomicKind(
        rewriter, loc, TypeRange{}, ValueRange{src, tmpUB}, ValueRange{resUB},
        op.getAtomicKind().value());
    if (!eltwiseOp.has_value()) {
      return op.emitError("not support block-sync atomic kind!!");
    }

    if (isUnsigned) {
      hivm::RoundMode rounding = mlir::utils::selectRoundMode<hivm::RoundMode>(
          rewriter.getF16Type(), getElementTypeOrSelf(dst));
      auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
      hivm::TypeFn typeFn = hivm::TypeFn::cast_unsigned;
      auto typeFnAttr = rewriter.getAttr<hivm::TypeFnAttr>(typeFn);
      resUB = castTo(rewriter, src.getLoc(), resUB, roundingAttr,
                     getElementTypeOrSelf(dst), typeFnAttr)
                  .getSingleDst();
    }

    // 4. store tmp to dst
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, resUB, dst);

    auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    unlockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

class AtomicRMWOpLowering : public OpRewritePattern<hivm::AtomicRMWOp> {
  using OpRewritePattern<hivm::AtomicRMWOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::AtomicRMWOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // If RMW don't have return args -> replace it to store
    if (op.getNumResults() == 0) {
      // convert hivm::AtomicRMWOp to hivm::StoreOp
      Value src = op.getSrc();
      Value dst = op.getDst();

      auto newStoreOp =
          rewriter.create<hivm::StoreOp>(loc, TypeRange(), src, dst);

      // Add atomic attr to hivm.store
      auto hsAtomicKind = op.getAtomicKind();
      newStoreOp.setAtomicKind(hsAtomicKind);

      rewriter.replaceOp(op, newStoreOp);
      return success();
    }
    // If RMW has return value we should always use software lock

    return decomposeEltwiseAtomic(op, rewriter, loc);
  }

  bool shouldCastOperation(hivm::AtomicRMWOp op) const {
    switch (op.getAtomicKind()) {
    case hivm::AtomicKind::ADD:
    case hivm::AtomicKind::MIN:
    case hivm::AtomicKind::MAX:
      return true;
    default:
      return false;
    }
  }

  Value processCastTo(PatternRewriter &rewriter, Location loc, Value val,
                      Type initType) const {
    if (initType.isInteger(8)) {
      auto roundingAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);

      val = castTo(rewriter, loc, val, roundingAttr, rewriter.getF16Type())
                .getSingleDst();
      val = castTo(rewriter, loc, val, roundingAttr, rewriter.getF32Type())
                .getSingleDst();
    } else if (initType.isBF16()) {
      auto roundingAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);

      val = castTo(rewriter, loc, val, roundingAttr, rewriter.getF32Type())
                .getSingleDst();
    }

    return val;
  }

  Value processCastFrom(PatternRewriter &rewriter, Location loc, Value val,
                        Type initType) const {
    if (initType.isInteger(8)) {
      auto truncAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::TRUNC);
      val = castTo(rewriter, loc, val, truncAttr, rewriter.getI32Type())
                .getSingleDst();

      auto truncOverflowAttr = rewriter.getAttr<hivm::RoundModeAttr>(
          hivm::RoundMode::TRUNCWITHOVERFLOW);
      val = castTo(rewriter, loc, val, truncOverflowAttr, rewriter.getI8Type())
                .getSingleDst();
    } else if (initType.isBF16()) {
      auto roundingAttr =
          rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);

      val = castTo(rewriter, loc, val, roundingAttr, rewriter.getBF16Type())
                .getSingleDst();
    }

    return val;
  }

  LogicalResult decomposeEltwiseAtomic(hivm::AtomicRMWOp op,
                                       PatternRewriter &rewriter,
                                       Location loc) const {
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    ::mlir::Value val = lockVar;
    Operation *lockDefOp = val.getDefiningOp();
    auto createLockOp = dyn_cast<hivm::CreateSyncBlockLockOp>(lockDefOp);
    if (createLockOp) {
        createLockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());
    }
    auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    lockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    // 2. create tmp memref alloc and load dst to tmp
    auto src = op.getSrc();
    auto tmpUB = createTmpBufferOrTensorWithTargetType(rewriter, loc,
                                                       op.getResults()[0]);
    rewriter.replaceAllUsesWith(op.getResults()[0], tmpUB);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    auto elementType = getElementTypeOrSelf(src);
    bool shouldCast = shouldCastOperation(op);
    if (shouldCast) {
      src = processCastTo(rewriter, op.getLoc(), src, elementType);
      tmpUB = processCastTo(rewriter, op.getLoc(), tmpUB, elementType);
    }

    // 3. do eltwise vv between src and tmp(and/or/xor)
    auto resUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src);
    auto eltwiseOp = createEltwiseOpByAtomicKind(
        rewriter, loc, TypeRange{}, ValueRange{src, tmpUB}, ValueRange{resUB},
        op.getAtomicKind());
    if (!eltwiseOp.has_value()) {
      return op.emitError("not support block-sync atomic kind!!");
    }

    if (shouldCast) {
      resUB = processCastFrom(rewriter, loc, resUB, elementType);
    }

    // 4. store tmp to dst
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, resUB, dst);

    auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    unlockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};

/// implement atomic cas in software way
/// e.g. hivm.hir.atomic_cas ins(%src0_ub, src1_ub) outs(%dst_gm) is converted
/// to
/// 1. %lock_var = create_sync_lock()
/// 2. sync_block_lock(%lock_var)
/// 3. %tmp0_ub = load(%dst_gm)
/// 4. %cond = vcmp(tmp0_ub, src0_ub)
/// 5. %tmp0_ub = vsel(%cond, src1_ub, tmp0_ub)
/// 6. %dst_gm = store(%tmp0_ub)
/// 7. sync_block_unlock(%lock_var)
class AtomicCasOpLowering : public OpRewritePattern<hivm::AtomicCasOp> {
  using OpRewritePattern<hivm::AtomicCasOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::AtomicCasOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    ::mlir::Value val = lockVar;
    Operation *lockDefOp = val.getDefiningOp();
    auto createLockOp = dyn_cast<hivm::CreateSyncBlockLockOp>(lockDefOp);
    if (createLockOp) {
        createLockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());
    }
    auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    lockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    // step1: load old val in gm to ub
    auto src0 = op.getSrc()[0];

    bool hasReturn = !op.getResults().empty();
    auto tmpUB = createTmpBufferOrTensorWithTargetType(
        rewriter, loc, hasReturn ? op.getResults()[0] : src0);

    auto dst = op.getDst();
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB);

    // step2: condition = vcmp(dst, expected_val)
    auto condUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src0,
                                                        rewriter.getI1Type());
    auto compareAttr =
        rewriter.getAttr<hivm::CompareModeAttr>(hivm::CompareMode::EQ);
    auto elemType = getElementTypeOrSelf(src0);
    auto src1 = op.getSrc()[1];
    hivm::RoundMode rounding = hivm::RoundMode::RINT;
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
    if (elemType.isInteger(8)) {
      src0 = castTo(rewriter, src0.getLoc(), src0, roundingAttr,
                    rewriter.getF16Type())
                 .getSingleDst();
      src1 = castTo(rewriter, src1.getLoc(), src1, roundingAttr,
                    rewriter.getF16Type())
                 .getSingleDst();
      tmpUB = castTo(rewriter, tmpUB.getLoc(), tmpUB, roundingAttr,
                     rewriter.getF16Type())
                  .getSingleDst();
    } else if (elemType.isBF16()) {
      src0 = castTo(rewriter, src0.getLoc(), src0, roundingAttr,
                    rewriter.getF32Type())
                 .getSingleDst();
      src1 = castTo(rewriter, src1.getLoc(), src1, roundingAttr,
                    rewriter.getF32Type())
                 .getSingleDst();
      tmpUB = castTo(rewriter, tmpUB.getLoc(), tmpUB, roundingAttr,
                     rewriter.getF32Type())
                  .getSingleDst();
    }
    rewriter.create<hivm::VCmpOp>(op.getLoc(), TypeRange(),
                                  ValueRange({tmpUB, src0}), Value(condUB),
                                  compareAttr);

    auto resUB = createTmpBufferOrTensorWithTargetType(rewriter, loc, src0);
    rewriter.create<hivm::VSelOp>(op.getLoc(), TypeRange(),
                                  ValueRange({condUB, src1, tmpUB}),
                                  ValueRange({resUB}), Value());
    if (elemType.isInteger(8)) {
      resUB = castTo(rewriter, resUB.getLoc(), resUB, roundingAttr,
                     rewriter.getI8Type())
                  .getSingleDst();
    } else if (elemType.isBF16()) {
      resUB = castTo(rewriter, resUB.getLoc(), resUB, roundingAttr,
                     rewriter.getBF16Type())
                  .getSingleDst();
    }

    // step3: store res_ub to dst
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, resUB, dst);

    auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    unlockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    if (hasReturn) {
      rewriter.replaceAllUsesWith(op.getResults()[0], tmpUB);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// implement atomic xchg in software way
/// e.g. hivm.hir.atomic_xchg ins(%src_ub) outs(%dst_gm) is converted to
/// 1. %lock_var = create_sync_lock()
/// 2. sync_block_lock(%lock_var)
/// 3. %tmp0_ub = load(%dst_gm)
/// 4. %dst_gm = store(%src_ub)
/// 5. %src_ub = copy(%tmp0_ub)
/// 7. sync_block_unlock(%lock_var)
class AtomicXchgOpLowering : public OpRewritePattern<hivm::AtomicXchgOp> {
  using OpRewritePattern<hivm::AtomicXchgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hivm::AtomicXchgOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lockVar = createSyncBlockLockVar(rewriter, op->getLoc());
    ::mlir::Value val = lockVar;
    Operation *lockDefOp = val.getDefiningOp();
    auto createLockOp = dyn_cast<hivm::CreateSyncBlockLockOp>(lockDefOp);
    if (createLockOp) {
        createLockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());
    }
    auto lockOp = rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
    lockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    auto src = op.getSrc();
    auto dst = op.getDst();
    auto mask = op.getMask();

    // step1: load old val in dst gm to ub
    bool hasReturn = !op.getResults().empty();
    auto tmpUB_dst = createTmpBufferOrTensorWithTargetType(
        rewriter, loc, hasReturn ? op.getResults()[0] : src);
    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, dst, tmpUB_dst);
    if (mask) {
      // step2: select according to the mask
      auto tmpUB_masked_dst =
          createTmpBufferOrTensorWithTargetType(rewriter, loc, src);
      rewriter.create<hivm::VSelOp>(loc, TypeRange{},
                                    ValueRange({mask, src, tmpUB_dst}),
                                    ValueRange({tmpUB_masked_dst}), Value());
      rewriter.create<hivm::VSelOp>(loc, TypeRange{},
                                    ValueRange({mask, tmpUB_dst, src}),
                                    ValueRange({src}), Value());
      // step3: copy/store the selected value
      rewriter.create<hivm::StoreOp>(loc, TypeRange{}, tmpUB_masked_dst, dst);
    } else {
      // step2: store new val to dst gm
      rewriter.create<hivm::StoreOp>(loc, TypeRange{}, src, dst);
      // step3: copy old val to src ub
      rewriter.create<hivm::CopyOp>(loc, TypeRange{}, tmpUB_dst, src);
    }

    auto unlockOp = rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
    unlockOp->setAttr(hivm::SyncBlockLockUnorderedAttr::name, rewriter.getUnitAttr());

    if (hasReturn) {
      rewriter.replaceAllUsesWith(op.getResults()[0], tmpUB_dst);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

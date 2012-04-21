//
//  Torch.m
//

#import "Torch.h"

@implementation Torch

- (void)require:(NSString *)file
{
    int ret = luaL_dofile(L, [[[NSBundle mainBundle] pathForResource:file ofType:@"lua"] UTF8String]);
    if (ret == 1) {
        NSLog(@"could not load invalid lua resource: %@\n", file);
    }
}

- (void)initialize
{
    // initialize Lua stack
    lua_executable_dir("./lua");
    L = lua_open();
    luaL_openlibs(L);
    
    // load Torch
    luaopen_libtorch(L);
    [self require:@"torch"];
    [self require:@"File"];
    [self require:@"CmdLine"];
    [self require:@"Tester"];

    // load dok
    [self require:@"dok.lua"];
    [self require:@"inline.lua"];
    
    // load nn
    luaopen_libnn(L);
    [self require:@"nn"];
    [self require:@"Module"];
    [self require:@"Concat"];
    [self require:@"Parallel"];
    [self require:@"Sequential"];
    [self require:@"Linear"];
    [self require:@"SparseLinear"];
    [self require:@"Reshape"];
    [self require:@"Select"];
    [self require:@"Narrow"];
    [self require:@"Replicate"];
    [self require:@"Copy"];
    [self require:@"Min"];
    [self require:@"Max"];
    [self require:@"Mean"];
    [self require:@"Sum"];
    [self require:@"CMul"];
    [self require:@"Mul"];
    [self require:@"Add"];
    [self require:@"CAddTable"];
    [self require:@"CDivTable"];
    [self require:@"CMulTable"];
    [self require:@"CSubTable"];
    [self require:@"Euclidean"];
    [self require:@"WeigthedEuclidean"];
    [self require:@"PairwiseDistance"];
    [self require:@"CosineDistance"];
    [self require:@"DotProduct"];
    [self require:@"Exp"];
    [self require:@"Log"];
    [self require:@"HardTanh"];
    [self require:@"LogSigmoid"];
    [self require:@"LogSoftMax"];
    [self require:@"Sigmoid"];
    [self require:@"SoftMax"];
    [self require:@"SoftMin"];
    [self require:@"SoftPlus"];
    [self require:@"SoftSign"];
    [self require:@"Tanh"];
    [self require:@"Abs"];
    [self require:@"Power"];
    [self require:@"Square"];
    [self require:@"Sqrt"];
    [self require:@"HardShrink"];
    [self require:@"SoftShrink"];
    [self require:@"Threshold"];
    [self require:@"LookupTable"];
    [self require:@"SpatialConvolution"];
    [self require:@"SpatialConvolutionMap"];
    [self require:@"SpatialSubSampling"];
    [self require:@"SpatialMaxPooling"];
    [self require:@"SpatialLPPooling"];
    [self require:@"TemporalConvolution"];
    [self require:@"TemporalSubSampling"];
    [self require:@"SpatialSubtractiveNormalization"];
    [self require:@"SpatialDivisiveNormalization"];
    [self require:@"SpatialContrastiveNormalization"];
    [self require:@"SpatialZeroPadding"];
    [self require:@"VolumetricConvolution"];
    [self require:@"ParallelTable"];
    [self require:@"ConcatTable"];
    [self require:@"SplitTable"];
    [self require:@"JoinTable"];
    [self require:@"CriterionTable"];
    [self require:@"Identity"];
    [self require:@"Criterion"];
    [self require:@"MSECriterion"];
    [self require:@"MarginCriterion"];
    [self require:@"AbsCriterion"];
    [self require:@"ClassCriterion"];
    [self require:@"DistKLDivCriterion"];
    [self require:@"MultiCriterion"];
    [self require:@"L1HingeCriterion"];
    [self require:@"HingeEmbeddingCriterion"];
    [self require:@"CosineEmbeddingCriterion"];
    [self require:@"MarginRankingCriterion"];
    [self require:@"MultiMarginCriterion"];
    [self require:@"MultiLabelMarginCriterion"];
    [self require:@"StochasticGradien"];
    [self require:@"Jacobian"];

    // load nnx
    luaopen_libnnx(L);
    [self require:@"nnx"];
    [self require:@"ConfusionMatrix.lua"];
    [self require:@"Logger.lua"];
    [self require:@"Probe.lua"];
    [self require:@"OmpModule.lua"];
    [self require:@"Minus.lua"];
    [self require:@"SpatialLinear.lua"];
    [self require:@"SpatialClassifier.lua"];
    [self require:@"SpatialPadding.lua"];
    [self require:@"SpatialNormalization.lua"];
    [self require:@"SpatialUpSampling.lua"];
    [self require:@"SpatialDownSampling.lua"];
    [self require:@"SpatialReSampling.lua"];
    [self require:@"SpatialRecursiveFovea.lua"];
    [self require:@"SpatialFovea.lua"];
    [self require:@"SpatialPyramid.lua"];
    [self require:@"SpatialGraph.lua"];
    [self require:@"SpatialMatching.lua"];
    [self require:@"SpatialMaxSampling.lua"];
    [self require:@"SpatialColorTransform.lua"];
    [self require:@"SpatialConvolutionSparse.lua"];
    [self require:@"SuperCriterion.lua"];
    [self require:@"SparseCriterion.lua"];
    [self require:@"DistNLLCriterion.lua"];
    [self require:@"KLDivCriterion.lua"];
    [self require:@"DistMarginCriterion.lua"];
    [self require:@"SpatialMSECriterion.lua"];
    [self require:@"SpatialClassNLLCriterion.lua"];
    [self require:@"SpatialSparseCriterion.lua"];
    [self require:@"Optimization.lua"];
    [self require:@"BatchOptimization.lua"];
    [self require:@"SNESOptimization.lua"];
    [self require:@"SGDOptimization.lua"];
    [self require:@"ASGDOptimization.lua"];
    [self require:@"LBFGSOptimization.lua"];
    [self require:@"CGOptimization.lua"];
    [self require:@"newCGOptimization.lua"];
    [self require:@"GeneticSGDOptimization.lua"];
    [self require:@"DiagHessian.lua"];
    [self require:@"Trainer.lua"];
    [self require:@"OnlineTrainer.lua"];
    [self require:@"BatchTrainer.lua"];
    [self require:@"Type.lua"];
    [self require:@"DataSet.lua"];
    [self require:@"DataList.lua"];
    [self require:@"DataSetLabelMe.lua"];

    // load image
    luaopen_libimage(L);
    [self require:@"image.lua"];

    // run user code
    [self require:@"main"];
    lua_getfield(L, LUA_GLOBALSINDEX, "initialize");
    lua_call(L, 0, 0);
    
    // done
    return;
}

@end

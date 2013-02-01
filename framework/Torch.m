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

    // load dok
    [self require:@"dok"];
    
    // load nn
    luaopen_libnn(L);
    [self require:@"nn"];

    // load nnx
    luaopen_libnnx(L);
    [self require:@"nnx"];

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

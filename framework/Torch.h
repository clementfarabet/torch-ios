//
//  Torch.h
//

#import <UIKit/UIKit.h>

#import <Torch/lua.h>
#import <Torch/TH/TH.h>
#import <Torch/luaT.h>
#import <Torch/lualib.h>
#import <Torch/lauxlib.h>

int luaopen_libtorch(lua_State *L);
int luaopen_libnn(lua_State *L);
int luaopen_libnnx(lua_State *L);
int luaopen_libimage(lua_State *L);

@interface Torch : NSObject
{
    lua_State *L;
}

- (void)initialize;
- (void)runMain:(NSString *)fileName inFolder:(NSString *)folderName;
- (void)loadFileWithName:(NSString *)filename inResourceFolder:(NSString *)folderName andLoadMethodName:(NSString *)methodName;
- (lua_State *)getLuaState;

@end

//
//  Torch.h
//

#import <UIKit/UIKit.h>

#import "lua.h"
#import "TH/TH.h"
#import "luaT.h"
#import "lualib.h"
#import "lauxlib.h"

int luaopen_libtorch(lua_State *L);
int luaopen_libnn(lua_State *L);
int luaopen_libnnx(lua_State *L);
int luaopen_libimage(lua_State *L);

@interface Torch : NSObject
{
    lua_State *L;
}

- (void)require:(NSString *)file;
- (void)initialize;

@end

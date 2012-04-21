Torch7 Library for iOS
======================

Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to an easy and fast scripting language (Lua) and a
underlying C implementation.

This package has been modified (or just hacked) to fully compile
Torch7 for iOS (iPad/iPhone).

Requirements
============
Torch7 needs to be installed prior to building the iOS
version. 'torch' needs to be available in the user's path.

Installation
============
Simply run:
$ ./generate_ios_framework

This will build all torch's libraries as static libs, and export them
in a single dir: framework/. The dir is ready to be included in
an iOS project.

Note: the libs are built for the ARMv7 arch, and paths to the XCode
frameworks are sort of hard coded in CMakeLists.txt, change them
if anything fails.

Running
=======
In your XCode/iOS code (Objective C), simply include the headers:

#import "lua.h"
#import "luaT.h"
#import "lualib.h"
#import "lauxlib.h"

and then define a function somewhere to initialize the Lua stack,
and load some Lua script (any Lua file can be loaded included as a
Resource in the XCode project):

- (void) initLua{
    // initialize Lua stack
    lua_executable_dir("./lua");
    lua_State *L = lua_open();
    luaL_openlibs(L);
    
    // load some lua, from resources
    NSString *mainpath = [[NSBundle mainBundle]
    pathForResource:@"main" ofType:@"lua"];

    // load code
    luaL_dofile(L, [mainpath UTF8String]);

    // run first function
    lua_getfield(L, LUA_GLOBALSINDEX, "initialize");
    lua_call(L, 0, 0);
    
    // done
    return;
}

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
an iOS project: it includes an example class to load Torch from within
your Objective C project.

Note: the libs are built for the ARMv7 arch, and paths to the XCode
frameworks are sort of hard coded in CMakeLists.txt, change them
if anything fails.

Running
=======
In your XCode/iOS code (Objective C), simply import the class
Torch.m/.h; include all the libs to the linker; and finally
add all the Lua files as resources. All you have left to do
is to define a main.lua file to keep going...

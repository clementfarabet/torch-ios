Torch7 Library for iOS
======================

Torch7 provides a Matlab-like environment for state-of-the-art machine
learning algorithms. It is easy to use and provides a very efficient
implementation, thanks to an easy and fast scripting language (Lua) and a
underlying C implementation.

This package has been modified (or just hacked) to fully compile
Torch7 for iOS (iPad/iPhone) for all architectures (armv7, armv7a, arm64, i386 (simulator), x86_64 (simulator))

Requirements
============

Torch7 needs to be installed prior to building the iOS
version. 'torch' needs to be available in the user's path.

I recommend doing the easy install if you have not installed Torch7.
http://torch.ch/docs/getting-started.html

Building The Framework
============
Simply run:
$ ./generate_ios_framework

This will build all torch's libraries as static libs, and export them
in a single dir: framework/. The dir is ready to be included in
an iOS project: it includes an example class to load Torch from within
your Objective C project.

For examples full examples that utilize this class (Torch.m) please see 
the ios_examples/ folder. More examples to come soon.

Running
=======
When creating your Objective-C project simply import the class
Torch.m/.h; include all the libs to the linker; add Torch.framework & Accelrate.framework
and add all the Lua files as resources. define YOUR_FILE.lua and add it as 
a resource. Run YOUR_FILE.lua using the method defined in Torch.h/.m
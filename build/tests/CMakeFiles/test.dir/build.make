# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Desktop/我的网盘/high-performance-compute-lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Desktop/我的网盘/high-performance-compute-lib/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/test.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test.dir/flags.make

tests/CMakeFiles/test.dir/kernels/test.cc.o: tests/CMakeFiles/test.dir/flags.make
tests/CMakeFiles/test.dir/kernels/test.cc.o: ../tests/kernels/test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Desktop/我的网盘/high-performance-compute-lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test.dir/kernels/test.cc.o"
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test.dir/kernels/test.cc.o -c /root/Desktop/我的网盘/high-performance-compute-lib/tests/kernels/test.cc

tests/CMakeFiles/test.dir/kernels/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test.dir/kernels/test.cc.i"
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/Desktop/我的网盘/high-performance-compute-lib/tests/kernels/test.cc > CMakeFiles/test.dir/kernels/test.cc.i

tests/CMakeFiles/test.dir/kernels/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test.dir/kernels/test.cc.s"
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/Desktop/我的网盘/high-performance-compute-lib/tests/kernels/test.cc -o CMakeFiles/test.dir/kernels/test.cc.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/kernels/test.cc.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

tests/test: tests/CMakeFiles/test.dir/kernels/test.cc.o
tests/test: tests/CMakeFiles/test.dir/build.make
tests/test: src/libkernels.a
tests/test: tests/CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/Desktop/我的网盘/high-performance-compute-lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test"
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test.dir/build: tests/test

.PHONY : tests/CMakeFiles/test.dir/build

tests/CMakeFiles/test.dir/clean:
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test.dir/clean

tests/CMakeFiles/test.dir/depend:
	cd /root/Desktop/我的网盘/high-performance-compute-lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Desktop/我的网盘/high-performance-compute-lib /root/Desktop/我的网盘/high-performance-compute-lib/tests /root/Desktop/我的网盘/high-performance-compute-lib/build /root/Desktop/我的网盘/high-performance-compute-lib/build/tests /root/Desktop/我的网盘/high-performance-compute-lib/build/tests/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test.dir/depend


## Installation</p>


### I. Requirements

####Software

 * A C++ compiler that supports C++11
 * [CMake](https://cmake.org)
 * [Boost](https://www.boost.org)
 * [DepSpawn](https://github.com/fraguela/depspawn)
 * [UPC++ 1.0](https://bitbucket.org/berkeleylab/upcxx/wiki/Home) The library has been built and tested on
 	* [Release 2021.3.0](https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2021.3.0.tar.gz)
 	* [Release 2022.9.0](https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2022.9.0.tar.gz)
 * Optional: a BLAS library if you want to compile the linear algebra benchmarks

####Hardware

 The library has been developed and tested in x86_64 systems, both under Linux and MacOS. Other architectures are not (yet) officially supported.

### II. Step by step procedure 


1. Install [DepSpawn](https://github.com/fraguela/depspawn) setting 
   * `DEPSPAWN_DUMMY_POOL` to `OFF`
   * `DEPSPAWN_FAST_START` to `ON`
   * `DEPSPAWN_SCALABLE_POOL` to `OFF`
   * `DEPSPAWN_USE_TBB` to `OFF`

2. Install [UPC++](https://bitbucket.org/berkeleylab/upcxx/wiki/Home)

3. Make sure the `bin` directoy of UPC++, located at `<upcxx_installation_directory>/bin`, is in your `$PATH`.

4. Unpack the UPC++ DepSpawn tarball (e.g. `upcxx_depspawn.tar.gz`)

		tar -xzf upcxx_depspawn.tar.gz
		cd upcxx_depspawn		
 or clone the project from its repository
 
	    git clone git@github.com:fraguela/upcxx_depspawn.git


5. Create the temporary directory where the project will be built and enter it :

		mkdir build && cd build

6. Generate the files for building the benchmaks and tests for the library in the format that you prefer (Visual Studio projects, nmake makefiles, UNIX makefiles, Mac Xcode projects, ...) using cmake.

    In this process you can use a graphical user interface for cmake such as `cmake-gui` in Unix/Mac OS X or `CMake-gui` in Windows, or a command-line interface such as `ccmake`. The process is explained here assuming this last possibility, as graphical user interfaces are not always available.
 
7. run `cmake -DCMAKE_PREFIX_PATH=<depspawn_installation_dir>/lib/cmake ..`
	
	 where `depspawn_installation_dir` is the root directory for the installation of [DepSpawn](https://github.com/fraguela/depspawn). This will generate the files for building UPC++ DepSpawn with the tool that cmake choses by default for your platform. Flag `-G` can be used to specify the kind of tool that you want to use. For example if you want to use Unix makefiles but they are not the default in you system, add the flag <tt>-G 'Unix Makefiles'</tt>

	 Run `cmake --help` for additional options and details.

8. Run `ccmake .` in order to configure your build.

9. Provide the values you wish for the variables that appear in the screen. The most relevant ones are:
	- `CMAKE_BUILD_TYPE` : String that specifies the build type. Its possible values are empty, Debug, Release, RelWithDebInfo and MinSizeRel.
	- `CMAKE_INSTALL_PREFIX` : Directory where UPC++ DepSpawn will be installed.

   The variables that begin with `DEPSPAWN` or `UPCXX_DEPSPAWN` change internal behaviors in the runtime of the library. Thus in principle they are of interest only to the developers of the library, although users can play with them to see how they impact in their applications. They are:
   
    - `DEPSPAWN_FAST_START` : If it is `ON`, when a thread that is spawning tasks detects that there are too many ready pending tasks waiting to be executed, it stops and executes one of them before resuming its current task.
    - `DEPSPAWN_PROFILE` : When it is `ON` the library gathers statistics about its internal operations.
   - `STACK_TILE` : This flag is not actually related to the library itself, but to some of its benchmarks. If this flag is `ON` these benchmarks will generate the tiles of their matrices in the stack memory; otherwise they will be allocated in the heap. Heap allocation allows the usage of larger tiles and will sometimes allow copying/moving tiles just by copying/moving the pointer to them.
   - `TILESIZE` : Also related to benchmarks but not to UPC++ DepSpawn itself. This is the tile size for several benchmarks.
   - `UPCXX_DEPSPAWN_ATOMIC_EXACT_MATCH` : Activates an alternative implementation for the Exact Match Mode (EMM). Only lightly tested.
   - `UPCXX_DEPSPAWN_AUTOMATIC_CACHE` : Enables this macro in the compilation of the tests and benchmarks provided. See its description in Section [Compiling applications](#compiling_applications).
   - `UPCXX_NETWORK` : For which GASNet-EX / UPC++ network to build the library and the binaries 
  
10. When you are done, press `c` to re-configure cmake with the new values.

11. Press `g` to generate the files that will be used to build the benchmarks and tests for the library and exit cmake.

12. The rest of this explanation assumes that UNIX makefiles were generated in the previous step. 

	Run `make` in order to build the benchmarks and the tests.  The degree of optimization, debugging information and assertions enabled depends on the value you chose for variable `CMAKE_BUILD_TYPE`.
	
    You can use the flag `-j` to speedup the building process. For example, `make -j4` will use 4 parallel processes, while `make -j` will use one parallel process per available hardware thread.

13. (Optionally) run `make check` in order to run the UPC++ DepSpawn tests. They are all run using a `upcxx-run -n 4` command

14. (Optionally) run `make benchmarks` in order to build the binaries in the `benchmarks` directory. Some of them require a BLAS library.

15. Run `make install` 

    This installs UPC++ DepSpawn under the directory you specified for the `CMAKE_INSTALL_PREFIX` variable. If you left it empty, the default base directories will be `/usr/local` in Unix and `c:/Program Files` in Windows. 

    The installation places the UPC++ DepSpawn library `upcxx_depspawn.a` in the subdirectory `lib` and the header files in the subdirectory `include/upcxx_depspawn`

16. You can remove the `upcxx_despawn` directory generated by the unpacking of the tarball or the cloning of the project repository.

## <a name="compiling_applications"></a> Compiling applications

 UPC++ DepSpawn applications are compiled the same way as any other UPC++ code. The sources should include the header file `upcxx_depspawn/upcxx_depspawn.h`, which requires having the parent directory of the UPC++ headers installation in the include path. Also, the final binary should link the `libupcxx_depspawn.a` library built. In addition, a number of compile time macros can control the function of UPC++ DepSpawn. Some macros are independent of those used during the compilation of the library, while others should be in accordance. Both groups are now described in turn.
 
 Free macros:
 
  - `UPCXX_DEPSPAWN_AUTOMATIC_CACHE` : If this macro is not defined, the user must manually control which UPC++ global pointers get cached and which ones do not by using as formal parameters of the parallel tasks the `cached_global_ptr` class template provided by UPC++ DepSpawn. When the macro is defined, the runtime automatically builds a `cached_global_ptr` associated to every `global_ptr` provided as argument to a task so that all remote accesses are cached. This macro has only been lightly tested.
 
  - `UPCXX_DEPSPAWN_NO_CACHE` is a macro that when defined disables the runtime software cache.

Dependent macros:

 - `DEPSPAWN_PROFILE` should be defined when compiling our application if we also chose it when compiling the library. Otherwise, many activities that happen in functions located in the header files of the library will not be reflected in the profiling.

  
 

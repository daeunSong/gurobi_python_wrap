cmake_minimum_required(VERSION 3.3)

project(solveLP C)

option(CXX "enable C++ compilation" ON)
if(CXX)
    enable_language(CXX)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(GUROBI REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

set(CMAKE_SHARED_MODULE_PREFIX "")

include_directories(${GUROBI_INCLUDE_DIRS})


# create the lib
#add_executable(${CMAKE_PROJECT_NAME} qp.cpp)
add_library(solveLP MODULE solveLP.cpp)
add_library(solveMIP MODULE solveMIP.cpp)

#if(CXX)
#    set(CMAKE_CXX_STANDARD 11)
#    target_link_libraries(${CMAKE_PROJECT_NAME} optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
#endif()

target_link_libraries(solveLP ${GUROBI_LIBRARY} ${PYTHON_LIBRARIES})
target_link_libraries(solveLP ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(solveLP PRIVATE ${PYTHON_INCLUDE_DIRS})

target_link_libraries(solveMIP ${GUROBI_LIBRARY} ${PYTHON_LIBRARIES})
target_link_libraries(solveMIP ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(solveMIP PRIVATE ${PYTHON_INCLUDE_DIRS})



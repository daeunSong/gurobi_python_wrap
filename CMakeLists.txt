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

set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}" )

# create the lib
#add_executable(${CMAKE_PROJECT_NAME} qp.cpp)
add_library(solveLP MODULE solveLP.cpp)
add_library(solveMIP MODULE solveMIP.cpp)


#~ target_link_libraries(solveLP ${PYTHON_LIBRARIES})
target_link_libraries(solveLP ${GUROBI_LIBRARIES}   ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(solveLP PRIVATE ${PYTHON_INCLUDE_DIRS})

target_link_libraries(solveMIP ${GUROBI_LIBRARY} ${PYTHON_LIBRARIES})
target_link_libraries(solveMIP ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(solveMIP PRIVATE ${PYTHON_INCLUDE_DIRS})



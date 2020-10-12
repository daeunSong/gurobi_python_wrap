cmake_minimum_required(VERSION 3.3)

#~ SET(PROJECT_ORG loco-3d)
SET(PROJECT_NAME qpp)
SET(PROJECT_DESCRIPTION "creatie and manipulate spline and bezier curves.")
SET(PROJECT_URL "https://github.com/${PROJECT_ORG}/${PROJECT_NAME}")

#~ project(qpp C)

option(CXX "enable C++ compilation" ON)
if(CXX)
    enable_language(CXX)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/python.cmake)
FINDPYTHON(3)
SEARCH_FOR_BOOST_PYTHON(REQUIRED)
#~ find_package(PythonLibs 2.7 REQUIRED)
#~ find_package(Boost COMPONENTS python REQUIRED)
find_package(GUROBI REQUIRED)
find_package(glpk REQUIRED)



set(CMAKE_SHARED_MODULE_PREFIX "")

include_directories(${GUROBI_INCLUDE_DIRS})
include_directories(${glpk_INCLUDE_DIRS})

set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}")

# create the lib
#add_executable(qpg qpg.cpp)
add_library(qpp MODULE qpp.cpp)


#~ target_link_libraries(solveLP ${PYTHON_LIBRARIES})
target_link_libraries(qpp ${PYTHON_LIBRARIES} ${GUROBI_LIBRARIES} ${Boost_LIBRARIES} ${glpk_LIBRARIES})
target_include_directories(qpp PRIVATE ${PYTHON_INCLUDE_DIRS})



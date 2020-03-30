#include "gurobi_c++.h"
#include <boost/numeric/ublas/matrix.hpp> 
#include <boost/python.hpp>

using namespace std;
using namespace boost::numeric::ublas;

int MAX = 1000;

int nonZeroCnt(matrix<double> A, int i){
    int cnt = 0;
    for (int j = 0; j < A.size2(); j ++){
        if (A(i,j) != 0.0)  cnt++;
    }
    return cnt;
}

void solveLP (const matrix<double> &c, const matrix<double> &A, const matrix<double> &b, 
              const matrix<double> &E, const matrix<double> &e)
{   
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        
        GRBVar cVars[c.size2()] = {};
        
        //add continuous variables
        for (int i = 0 ; i < c.size2(); i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c(0,i), GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();
        
        //// equality constraints
        if (E.size1() > 0){
            for (int i = 0; i < E.size1(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(E, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < E.size2(); j++){
                    if (E(i,j) != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[lenIdx] = {};
                double coeff[lenIdx] = {};
                for (int j = 0; j < lenIdx; j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E(i,idx[j]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e(0,i));
                //cout << expr << endl;
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size1() > 0){
            for (int i = 0; i < A.size1(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(A, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < A.size2(); j++){
                    if (A(i,j) != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[lenIdx] = {};
                double coeff[lenIdx] = {};
                for (int j = 0; j < lenIdx; j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A(i,idx[j]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b(0,i));
                //cout << expr << endl;
            }
            
        }
        model.update();
        //model.modelSense = grb.GRB.MINIMIZE
        //model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model.optimize();
        
        float result [c.size2()] = {};
        for (int i = 0; i < c.size2(); i++){
            result[i] = cVars[i].get(GRB_DoubleAttr_X);
            cout << result[i] << endl;
        }
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    }
    
}


BOOST_PYTHON_MODULE(solveLP) {
    // An established convention for using boost.python.
    using namespace boost::python;

    // Expose the function hello().
    def("solveLP", solveLP);
}

 
int
main(int   argc,
     char *argv[])
{    
    double MATRIX_A[3][3] = {{-1.0, 0.0, 0.0}, {-0.0,-1.0, 0.0}, {0.0,0.0,-1.0}};
    double MATRIX_b[1][3] = {{-1.0,2.0,3.0}};
    double MATRIX_C[1][3] = {{1.0,1.0,1.0}};
    double MATRIX_d[1][1] = {{1.0}};
    
    ////////temp initialization for test
    matrix<double> A (3,3); 
    matrix<double> b (1,3); 
    matrix<double> C (1,3); 
    matrix<double> d (1,1); 

    for (int i = 0; i < A.size1(); i++)
        for (int j = 0; j < A.size2(); j++)
            A(i,j) = MATRIX_A[i][j];
            
    for (int i = 0; i < b.size1(); i++)
        for (int j = 0; j < b.size2(); j++)
            b(i,j) = MATRIX_b[i][j];
                
    for (int i = 0; i < C.size1(); i++)
        for (int j = 0; j < C.size2(); j++)
            C(i,j) = MATRIX_C[i][j];
            
    for (int i = 0; i < d.size1(); i++)
        for (int j = 0; j < d.size2(); j++)
            d(i,j) = MATRIX_d[i][j];
            

    //////////////////
    
    cout << "solveLP" << endl;
    solveLP(b, A, b, C, d);
    //cout << "solveMIP" << endl;
    //solveMIP(b, A, b, C, d);


    return 0;
}


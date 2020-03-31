#include "gurobi_c++.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>

using namespace std;
using namespace boost::python;

int MAX = 1000;

int nonZeroCnt(list& A, int i){
    int cnt = 0;
    for (int j = 0; j < len(A[i]); j ++){
        if (extract<double>(A[i][j]) != 0.0)  cnt++;
    }
    return cnt;
}

void solveLP (list& c, list& A, list& b, list& E, list& e)
{ 
    
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        
        GRBVar cVars[len(c)] = {};
        
        //add continuous variables
        for (int i = 0 ; i < len(c); i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, extract<double>(c[i]), GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();        
        
        //// equality constraints
        if (len(E) > 0){
            for (int i = 0; i < len(E); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(E, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < len(E[i]); j++){
                    if (extract<double>(E[i][j]) != 0.0){
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
                    coeff[j] = extract<double>(E[i][idx[j]]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                double ee = extract<double>(e[i]);
                model.addConstr(expr == ee);
                //cout << expr << endl;
            }
        }
        model.update();
        
        //// inequality constraints
        if (len(A) > 0){
            for (int i = 0; i < len(A); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(A, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < len(A[i]); j++){
                    if (extract<double>(A[i][j]) != 0.0){
                        cout << i <<"," << j << endl;
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
                    coeff[j] = extract<double>(A[i][idx[j]]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                double bb = extract<double>(b[i]);
                model.addConstr(expr <= bb);
                //cout << expr << endl;
            }
            
        }
        model.update();
        //model.modelSense = grb.GRB.MINIMIZE
        //model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model.optimize();
        
        
        float result [len(c)] = {};
        for (int i = 0; i < len(c); i++){
            result[i] = cVars[i].get(GRB_DoubleAttr_X);
            cout << result[i] << endl;
        }
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }
    
}
  
  
void solveMIP (list& c, list& A, list& b, list& E, list& e)
{
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        
        GRBVar cVars[len(c)] = {};
               
        //add continuous variables
        for (int i = 0 ; i < len(c); i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, extract<double>(c[i]), GRB_CONTINUOUS, "x");
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();        
        
        //// equality constraints
        if (len(E) > 0){
            for (int i = 0; i < len(E); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(E, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < len(E[i]); j++){
                    if (extract<double>(E[i][j]) != 0.0){
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
                    coeff[j] = extract<double>(E[i][idx[j]]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                double ee = extract<double>(e[i]);
                model.addConstr(expr == ee);
                //cout << expr << endl;
            }
        }
        model.update();
        
        //// inequality constraints
        if (len(A) > 0){
            for (int i = 0; i < len(A); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(A, i);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < len(A[i]); j++){
                    if (extract<double>(A[i][j]) != 0.0){
                        cout << i <<"," << j << endl;
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
                    coeff[j] = extract<double>(A[i][idx[j]]);
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                double bb = extract<double>(b[i]);
                model.addConstr(expr <= bb);
                //cout << expr << endl;
            }
            
        }
        model.update();
        
        int slackIndices[len(c)] = {};
        int numSlackVar = 0;
        for (int i = 0; i < len(c); i++){
            if (extract<double>(c[i]) > 0){
                slackIndices[numSlackVar] = i;
                numSlackVar++;
            }
        }
        
        int numX = model.get(GRB_IntAttr_NumVars);
        
        // add boolean variables
        GRBVar bVars[numSlackVar] = {};
        for (int i = 0; i < numSlackVar; i++)
            bVars[i] = model.addVar(0, 1, 0, GRB_BINARY, "y");
        model.update(); 
        
        GRBVar* y = 0;
        y = model.getVars();

        // inequality
        GRBLinExpr expr = 0;
        for (int i = 0; i < numSlackVar; i++ ){
            expr += 1.0 * x[slackIndices[i]];
            expr += -100.0 * y[i+numX];
            model.addConstr(expr <= 0);
        }

        // equality
        int varIndices[MAX] = {};
        double previousL = 0.0;
        int k = 0;
        expr = 0;
        for (int i = 0; i < numSlackVar; i++ ){
            if (i != 0 && slackIndices[i] - previousL > 2){
                expr = 0;
                for (int j = 0 ; j < k ; j ++)
                    expr += y[varIndices[k]];
                model.addConstr(expr == k-1);
                delete varIndices; int varIndices[MAX] = {}; k =0;
                varIndices[k] = i+numX; k++;
            }
            else if (slackIndices[i] != 0){
                varIndices[k] = i+numX;   
                k++;
            }
            previousL = slackIndices[i];
        }
        
        if (k > 1){
            expr = 0;
            for (int i = 0; i < k; i++){
                expr += y[varIndices[i]];
            }
            model.addConstr(expr == k-1);
        }
        model.update();    
        
        expr = 0;
        for (int i = 0; i <numSlackVar; i++)
            expr += y[i+numX];
        
        model.setObjective(expr,GRB_MINIMIZE);
        model.optimize();
        
        float result [len(c)] = {};
        for (int i = 0; i < len(c); i++){
            result[i] = cVars[i].get(GRB_DoubleAttr_X);
            cout << result[i] << endl;
        }
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    }
    
    
}


BOOST_PYTHON_MODULE(qpg) {
    // An established convention for using boost.python.
    using namespace boost::python;

    // Expose the function hello().
    def("solveLP", solveLP);
    def("solveMIP", solveMIP);
}

 
int
main(int   argc,
     char *argv[])
{    
    Py_Initialize();
    
    double MATRIX_A[3][3] = {{-1.0, 0.0, 0.0}, {-0.0,-1.0, 0.0}, {0.0,0.0,-1.0}};
    double MATRIX_b[3] = {-1.0,2.0,3.0};
    double MATRIX_C[1][3] = {{1.0,1.0,1.0}};
    double MATRIX_d[1] = {1.0};
    
    list A;
    list b;
    list C;
    list d;
    
    for (int i = 0; i < 3; i ++){
        list tmp;
        for (int j = 0; j < 3; j++){
            tmp.append(MATRIX_A[i][j]);
        }
        A.append(tmp);
    }
    
    for (int i = 0; i < 3; i++){
        b.append(MATRIX_b[i]);
    }
    
    for (int i = 0; i < 1; i ++){
        list tmp;
        for (int j = 0; j < 3; j++){
            tmp.append(MATRIX_C[i][j]);
        }
        C.append(tmp);
    }
    
    for (int i = 0; i < 1; i++){
        d.append(MATRIX_d[i]);
    }
    
    cout << "solveLP" << endl;
    solveLP(b, A, b, C, d);
    cout << "solveMIP" << endl;
    solveMIP(b, A, b, C, d);


    return 0;
}


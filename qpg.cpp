#include "gurobi_c++.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>

using namespace std;
using namespace boost::python;

int MAX = 1000;

int nonZeroCnt(double* A, int len){
    int cnt = 0;
    for (int j = 0; j < len; j ++){
        if (A[j] != 0.0)  cnt++;
    }
    return cnt;
}

list solveLP (list& c_, list& A_, list& b_, list& E_, list& e_)
{ 
    double c[len(c_)] = {};
    double A[len(A_)][len(A_[0])] = {};
    double b[len(b_)] = {};
    double E[len(E_)][len(E_[0])] = {};
    double e[len(e_)] = {};
    
    int lenc = len(c_);
    int lenA1 = len(A_);
    int lenA2 = len(A_[0]);
    int lenb = len(b_);
    int lenE1 = len(E_);
    int lenE2 = len(E_[0]);
    int lene = len(e_);
    
    for (int i = 0; i < lenc; i++)
        c[i] = extract<double>(c_[i]);
    
    for (int i = 0; i < lenA1; i++){
        for (int j = 0; j < lenA2; j++)
            A[i][j] = extract<double>(A_[i][j]);
    }
        
    for (int i = 0; i < lenb; i++)
        b[i] = extract<double>(b_[i]);

    for (int i = 0; i < lenE1; i++){
        for (int j = 0; j < lenE2; j++)
            E[i][j] = extract<double>(E_[i][j]);
    }
        
    for (int i = 0; i < lene; i++)
        e[i] = extract<double>(e_[i]);
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        GRBVar cVars[lenc] = {};
        
        //add continuous variables
        for (int i = 0 ; i < lenc; i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();        
        
        //// equality constraints
        if (lenE1 > 0){
            for (int i = 0; i < lenE1; i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(E[i], lenE2);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < lenE2; j++){
                    if (E[i][j] != 0.0){
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
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
            }
        }
        model.update();
        
        //// inequality constraints
        if (lenA1 > 0){
            for (int i = 0; i < lenA1; i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(A[i], lenA2);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < lenA2; j++){
                    if (A[i][j] != 0.0){
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
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
            }
            
        }
        model.update();
        //model.modelSense = grb.GRB.MINIMIZE
        //model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model.optimize();
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        
        list result;
        result.append(time);
        for (int i = 0; i < lenc; i++){
            result.append(cVars[i].get(GRB_DoubleAttr_X));
            //cout << result[i] << endl;
        }
        
        return result;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }
    
}
  
  
list solveMIP (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    double c[len(c_)] = {};
    double A[len(A_)][len(A_[0])] = {};
    double b[len(b_)] = {};
    double E[len(E_)][len(E_[0])] = {};
    double e[len(e_)] = {};

    int lenc = len(c_);
    int lenA1 = len(A_);
    int lenA2 = len(A_[0]);
    int lenb = len(b_);
    int lenE1 = len(E_);
    int lenE2 = len(E_[0]);
    int lene = len(e_);
    
    for (int i = 0; i < lenc; i++)
        c[i] = extract<double>(c_[i]);
    
    for (int i = 0; i < lenA1; i++){
        for (int j = 0; j < lenA2; j++)
            A[i][j] = extract<double>(A_[i][j]);
    }
        
    for (int i = 0; i < lenb; i++)
        b[i] = extract<double>(b_[i]);

    for (int i = 0; i < lenE1; i++){
        for (int j = 0; j < lenE2; j++)
            E[i][j] = extract<double>(E_[i][j]);
    }
        
    for (int i = 0; i < lene; i++)
        e[i] = extract<double>(e_[i]);
        
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        GRBVar cVars[lenc] = {};
               
        //add continuous variables
        for (int i = 0 ; i < lenc; i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();        
        
        //// equality constraints
        if (lenE1 > 0){
            for (int i = 0; i < lenE1; i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(E[i], lenE2);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < lenE2; j++){
                    if (E[i][j] != 0.0){
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
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
            }
        }
        model.update();
        
        //// inequality constraints
        if (lenA1 > 0){
            for (int i = 0; i < lenA1; i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                int lenIdx = nonZeroCnt(A[i], lenA2);
                int idx[lenIdx] = {}; int k = 0;
                for (int j = 0; j < lenA2; j++){
                    if (A[i][j] != 0.0){
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
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, lenIdx);
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
            }
            
        }
        model.update();
        
        int slackIndices[lenc] = {};
        int numSlackVar = 0;
        for (int i = 0; i < lenc; i++){
            if (c[i] > 0){
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
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        
        list result;
        result.append(time);
        for (int i = 0; i < lenc; i++){
            result.append(cVars[i].get(GRB_DoubleAttr_X));
            //cout << result[i] << endl;
        }
        
        return result;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
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
    list result;
    
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
    
    //cout << "solveLP" << endl;
    //result = solveLP(b, A, b, C, d);
    //cout << "solveMIP" << endl;
    //result = solveMIP(b, A, b, C, d);


    return 0;
}


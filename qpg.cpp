#include "gurobi_c++.h"
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include <glpk.h>
//#include <algorithm>
//#include <Eigen/Dense>

using namespace std;
using namespace boost::python;

int MAX = 2000;
static const double UNBOUNDED_UP = 100000.;
static const double UNBOUNDED_DOWN = -100000.;


int nonZeroCnt(double* A, int len){
    int cnt = 0;
    for (int j = 0; j < len; j ++){
        if (A[j] != 0.0)  cnt++;
    }
    return cnt;
}

//int getType(const VectorXd& mib, const VectorXd& mab, const int i) {
  //const double& mibV = mib(i);
  //const double& mabV = mab(i);
  //int type = GLP_FR;
  //if (mibV > UNBOUNDED_DOWN && mabV < UNBOUNDED_UP)
    //type = GLP_DB;
  //else if (mibV > UNBOUNDED_DOWN)
    //type = GLP_LO;
  //else if (mabV < UNBOUNDED_UP)
    //type = GLP_UP;
  //return type;
//}


/*int solveglpk(const VectorXd& g0, const MatrixXd& CE, const VectorXd& ce0, const MatrixXd& CI, const VectorXd& ci0,
              solvers::Cref_vectorX minBounds, solvers::Cref_vectorX maxBounds, VectorXd& x, double& cost) {*/
list solveglpk(list& c_, list& A_, list& b_, list& E_, list& e_){

    double g0[len(c_)] = {};
    double CI[len(A_)][len(A_[0])] = {};
    double ci0[len(b_)] = {};
    double CE[len(E_)][len(E_[0])] = {};
    double ce0[len(e_)] = {};
    
    int leng = len(c_);
    int lenI1 = len(A_);
    int lenI2 = len(A_[0]);
    int leni = len(b_);
    int lenE1 = len(E_);
    int lenE2 = len(E_[0]);
    int lene = len(e_);
    
    for (int i = 0; i < leng; i++)
        g0[i] = extract<double>(c_[i]);
    
    for (int i = 0; i < lenI1; i++){
        for (int j = 0; j < lenI2; j++)
            CI[i][j] = extract<double>(A_[i][j]);
    }
        
    for (int i = 0; i < leni; i++)
        ci0[i] = extract<double>(b_[i]);

    for (int i = 0; i < lenE1; i++){
        for (int j = 0; j < lenE2; j++)
            CE[i][j] = extract<double>(E_[i][j]);
    }
        
    for (int i = 0; i < lene; i++)
        ce0[i] = extract<double>(e_[i]);   
    
    const clock_t begin_time = clock();
    
    double cost;
    glp_smcp opts;
    glp_init_smcp(&opts);
    opts.msg_lev = GLP_MSG_OFF;
    glp_prob* lp;
    int ia[1 + 20000];                // Row indices of each element
    int ja[1 + 20000];                // column indices of each element
    double ar[1 + 20000];             // numerical values of corresponding elements
    lp = glp_create_prob();           // creates a problem object
    glp_set_prob_name(lp, "sample");  // assigns a symbolic name to the problem object
    glp_set_obj_dir(lp, GLP_MIN);     // calls the routine glp_set_obj_dir to set the
                                    // omptimization direction flag,
                                    // where GLP_MAX means maximization

    // ROWS
    const int numEqConstraints = lenE1;
    const int numIneqConstraints = lenI1;
    const int num_constraints_total = numEqConstraints + numIneqConstraints;
    glp_add_rows(lp, num_constraints_total);
    int idrow = 1;  // adds three rows to the problem object
    int idcol = 1;
    int idConsMat = 1;
    int xsize = 0;
    if (lenE1 != 0) xsize = lenE2;
    if (lenI1 != 0) xsize = lenI2;
    //int xsize = (int)(x.size());
    
    for (int i = 0; i < numIneqConstraints; ++i, ++idrow) {
        glp_set_row_bnds(lp, idrow, GLP_UP, 0.0, ci0[i]);
        for (int j = 0; j < xsize; ++j, ++idcol) {
            if (CI[i][j] != 0.) {
                ia[idConsMat] = idrow, ja[idConsMat] = idcol, ar[idConsMat] = CI[i][j]; // a[1,1] = 1 
                ++idConsMat;
            }
        }
        idcol = 1;
    }
    for (int i = 0; i < numEqConstraints; ++i, ++idrow) {
        glp_set_row_bnds(lp, idrow, GLP_FX, ce0[i], ce0[i]);
        for (int j = 0; j < xsize; ++j, ++idcol) {
            if (CE[i][j] != 0.) {
                ia[idConsMat] = idrow, ja[idConsMat] = idcol, ar[idConsMat] = CE[i][j]; // a[1,1] = 1 
                ++idConsMat;
            }
        }
        idcol = 1;
    }

    // COLUMNS
    glp_add_cols(lp, xsize);
    //VectorXd miB = minBounds.size() > 0 ? minBounds : VectorXd::Ones(xsize) * UNBOUNDED_DOWN;
    //VectorXd maB = maxBounds.size() > 0 ? maxBounds : VectorXd::Ones(xsize) * UNBOUNDED_UP;
    for (int i = 0; i < xsize; ++i, ++idcol) {
        glp_set_col_bnds(lp, idcol, GLP_FR, UNBOUNDED_DOWN, UNBOUNDED_UP);
        glp_set_obj_coef(lp, idcol, g0[i]);
    }
    glp_load_matrix(lp, idConsMat - 1, ia, ja, ar);

    int res = glp_simplex(lp, &opts);
    const clock_t end_time = clock();
    res = glp_get_status(lp);
    double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
    list result;
    result.append(time);
    if (res == GLP_OPT) {
        cost = glp_get_obj_val(lp);  // obtains a computed value of the objective function
        idrow = 1;
        for (int i = 0; i < xsize; ++i, ++idrow) result.append(glp_get_col_prim(lp, idrow));
    }
    glp_delete_prob(lp);  // calls the routine glp_delete_prob, which frees all the memory
    glp_free_env();
    
    //for (int i = 0; i < xsize; ++i)
        //cout << extract<double>(result[i]) << endl;
    
    return result;
}


list solveLP (list& c_, list& A_, list& b_, list& E_, list& e_)
{ 
    double c[len(c_)];
    double A[len(A_)][len(A_[0])];
    double b[len(b_)];
    double E[len(E_)][len(E_[0])];
    double e[len(e_)];
    
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
        
        GRBVar cVars[lenc];
        
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
                int idx[lenIdx]; int k = 0;
                for (int j = 0; j < lenE2; j++){
                    if (E[i][j] != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[lenIdx];
                double coeff[lenIdx];
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
                int idx[lenIdx]; int k = 0;
                for (int j = 0; j < lenA2; j++){
                    if (A[i][j] != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[lenIdx];
                double coeff[lenIdx];
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
            //cout << extract<double>(result[i]) << endl;
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
    double c[len(c_)];
    double A[len(A_)][len(A_[0])];
    double b[len(b_)];
    double E[len(E_)][len(E_[0])];
    double e[len(e_)];

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
        
        GRBVar cVars[lenc];
               
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
                int idx[lenIdx]; int k = 0;
                for (int j = 0; j < lenE2; j++){
                    if (E[i][j] != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[lenIdx];
                double coeff[lenIdx];
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
                int idx[lenIdx]; int k = 0;
                for (int j = 0; j < lenA2; j++){
                    if (A[i][j] != 0.0){
                        idx[k] = j;
                        k++;
                    }
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[lenIdx];
                double coeff[lenIdx] ;
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
        
        int slackIndices[lenc];
        int numSlackVar = 0;
        for (int i = 0; i < lenc; i++){
            if (c[i] > 0){
                slackIndices[numSlackVar] = i;
                numSlackVar++;
            }
        }
        
        int numX = model.get(GRB_IntAttr_NumVars);
        
        // add boolean variables
        GRBVar bVars[numSlackVar];
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
        int varIndices[MAX];
        double previousL = 0.0;
        int k = 0;
        expr = 0;
        cout << numSlackVar << endl;  
        for (int i = 0; i < numSlackVar; i++ ){
            if (i != 0 && slackIndices[i] - previousL > 2){
                expr = 0;
                for (int j = 0 ; j < k ; j ++)
                    expr += y[varIndices[k]];
                model.addConstr(expr == k-1);
                //delete varIndices; int varIndices[MAX]; k =0;
                varIndices[MAX]={}; k =0;
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
            //cout << extract<double>(result[i]) << endl;
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
    def("solveglpk", solveglpk);
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
    
    cout << "solveLP" << endl;
    result = solveLP(b, A, b, C, d);
    cout << "solveMIP" << endl;
    result = solveMIP(b, A, b, C, d);
    cout << "solveglpk" << endl;
    result = solveglpk(b, A, b, C, d);

    return 0;
}


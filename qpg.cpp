#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/extract.hpp>
#include "gurobi_c++.h"
#include <glpk.h>
#include <vector>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace boost::python;

static const double UNBOUNDED_UP = 100000.;
static const double UNBOUNDED_DOWN = -100000.;

typedef struct ResultData
{
    list x;
    double time;
} result;

int nonZeroCnt(double* A, int len)
{
    int cnt = 0;
    for (int j = 0; j < len; j ++)
        if (A[j] != 0.0)  cnt++;
    
    return cnt;
}

void massadd (list& l, vector<double>& v)
{
    for (int i = 0; i < len(l); i++)
        v.push_back(extract<double>(l[i]));
}

void massadd2 (list& l, vector< vector<double> >& v)
{
    for (int i = 0; i < len(l); i++)
    {
        vector<double> tmp;
        for (int j = 0; j < len(l[0]); j++)
            tmp.push_back(extract<double>(l[i][j]));
        v.push_back(tmp);
    }
}

list solveglpk (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    // python::list to std::vector 
    vector<double> g0;
    massadd(c_, g0);
    
    vector< vector<double> > CI;
    massadd2(A_, CI);
    
    vector<double> ci0;
    massadd(b_, ci0);
    
    vector< vector<double> > CE;
    massadd2(E_, CE);
    
    vector<double> ce0;
    massadd(e_, ce0);
    
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
    const int numEqConstraints = CE.size();
    const int numIneqConstraints = CI.size();
    const int num_constraints_total = numEqConstraints + numIneqConstraints;
    glp_add_rows(lp, num_constraints_total);
    int idrow = 1;  // adds three rows to the problem object
    int idcol = 1;
    int idConsMat = 1;
    int xsize = 0;
    if (CE.size() != 0) xsize = CE[0].size();
    if (CI.size() != 0) xsize = CI[0].size();
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
    
    // for (int i = 0; i < xsize; ++i)
        // cout << extract<double>(result[i]) << endl;
    
    return result;
}


list solveLP (list& c_, list& A_, list& b_, list& E_, list& e_)
{   
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        GRBVar cVars[c.size()];
        
        //add continuous variables
        for (int i = 0 ; i < c.size(); i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();        
        
        //// equality constraints
        if (E.size() > 0){
            for (int i = 0; i < E.size(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++){
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
                idx.clear();
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size() > 0){
            for (int i = 0; i < A.size(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++){
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
                idx.clear();
            }
            
        }
        model.update();

        GRBQuadExpr expr = 0;
        // slack variables sum
        for (int i = 0; i <c.size(); i++)
            expr += x[i]*c[i]; //+numX
        model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model.optimize();
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list result;
        result.append(time);
        for (int i = 0; i < c.size(); i++){
            result.append(cVars[i].get(GRB_DoubleAttr_X));
            // cout << extract<double>(result[i]) << endl;
        }
        
        return result;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }
    
}
  

list solveLP_cost (list& c_, list& A_, list& b_, list& E_, list& e_, int nVarEnd, list& goal_)
{   
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    vector<double> goal;
    massadd(goal_,goal);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        int numC = c.size();
        GRBVar cVars[numC];
               
        //add continuous variables
        for (int i = 0 ; i < numC; i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
            if (i == numC-nVarEnd || i == numC-nVarEnd+1 || i == numC-nVarEnd+2)    // for the cost function
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars(); 
        
        //// equality constraints
        if (E.size() > 0){
            for (int i = 0; i < E.size(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++){
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
                idx.clear();
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size() > 0){
            for (int i = 0; i < A.size(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++){
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
                idx.clear();
            }
            
        }
        
        model.update();    
        
        GRBQuadExpr expr = 0;
        // slack variables sum
        for (int i = 0; i <numC; i++)
            expr += x[i]*c[i]; //+numX

        // distance cost function
        GRBLinExpr cx_end_diff = x[numC-nVarEnd]   - goal[0];
        GRBLinExpr cy_end_diff = x[numC-nVarEnd+1] - goal[1];
        GRBLinExpr cz_end_diff = x[numC-nVarEnd+2] - goal[2];
        expr += cx_end_diff * cx_end_diff + cy_end_diff * cy_end_diff + cz_end_diff * cz_end_diff;
        
        model.setObjective(expr,GRB_MINIMIZE);
        model.optimize();
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list result;
        result.append(time);
        for (int i = 0; i < c.size(); i++){
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

result solveLP_cost_init (list& c_, list& A_, list& b_, list& E_, list& e_, int nVarEnd, list& goal_, double weight)
{   
    result res;
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    vector<double> goal;
    massadd(goal_,goal);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        int numC = c.size();
        GRBVar cVars[numC];
               
        //add continuous variables
        for (int i = 0 ; i < numC; i++){
            if (i == numC-nVarEnd || i == numC-nVarEnd+1 || i == numC-nVarEnd+2)    // for the cost function
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
            else 
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars(); 
        
        //// equality constraints
        if (E.size() > 0){
            for (int i = 0; i < E.size(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++){
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
                idx.clear();
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size() > 0){
            for (int i = 0; i < A.size(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++){
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
                idx.clear();
            }
            
        }
        
        model.update();    
        
        GRBQuadExpr expr = 0;
        // slack variables sum
        for (int i = 0; i <numC; i++)
            expr += x[i]*c[i]; //+numX

        // distance cost function
        GRBLinExpr cx_end_diff = x[numC-nVarEnd]   - goal[0];
        GRBLinExpr cy_end_diff = x[numC-nVarEnd+1] - goal[1];
        GRBLinExpr cz_end_diff = x[numC-nVarEnd+2] - goal[2];
        expr += (cx_end_diff * cx_end_diff + cy_end_diff * cy_end_diff + cz_end_diff * cz_end_diff) * weight;
        
        model.setObjective(expr,GRB_MINIMIZE);
        model.optimize();
        model.write("model.mps");
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list r;
        res.time = time;
        for (int i = 0; i < c.size(); i++){
            r.append(cVars[i].get(GRB_DoubleAttr_X));
            //cout << extract<double>(x[i]) << endl;
        }
        res.x = r;
        
        return res;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }
    
}

result solveLP_cost_iter (list& c_, list& A_, list& b_, list& E_, list& e_, int nVarEnd, list& goal_, double weight)
{   
    result res;
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    vector<double> goal;
    massadd(goal_,goal);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel* model = new GRBModel(env, "model.mps");
        model->getEnv().set(GRB_IntParam_OutputFlag, 0);

        int numC = c.size();
        GRBVar cVars[numC];

        GRBVar* x = 0;
        x = model->getVars(); 
               
        //modify variables
        for (int i = 0 ; i < numC; i++){
            if (i == numC-nVarEnd || i == numC-nVarEnd+1 || i == numC-nVarEnd+2)    // for the cost function
                x[i].set(GRB_DoubleAttr_Obj, 1.);
            else
                x[i].set(GRB_DoubleAttr_Obj, c[i]);
            cVars[i] = x[i];
        }
        model->update(); 
        
        GRBQuadExpr expr = 0;
        // slack variables sum
        for (int i = 0; i <numC; i++)
            expr += x[i]*c[i]; //+numX

        // distance cost function
        GRBLinExpr cx_end_diff = x[numC-nVarEnd]   - goal[0];
        GRBLinExpr cy_end_diff = x[numC-nVarEnd+1] - goal[1];
        GRBLinExpr cz_end_diff = x[numC-nVarEnd+2] - goal[2];
        expr += (cx_end_diff * cx_end_diff + cy_end_diff * cy_end_diff + cz_end_diff * cz_end_diff) * weight;
        
        model->setObjective(expr,GRB_MINIMIZE);
        model->optimize();
        model->write("model.mps");
        
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list r;
        res.time = time;
        for (int i = 0; i < c.size(); i++){
            r.append(cVars[i].get(GRB_DoubleAttr_X));
            //cout << extract<double>(x[i]) << endl;
        }
        res.x = r;
        
        return res;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }
    
}  
  
list solveMIP (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    
    //random gen
    srand((unsigned int)time(NULL));
    
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        GRBVar cVars[c.size()];
        
        //add continuous variables
        for (int i = 0 ; i < c.size(); i++){
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();  
        int numX = model.get(GRB_IntAttr_NumVars);        
        
        //// equality constraints
        if (E.size() > 0){
            for (int i = 0; i < E.size(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++){
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
                idx.clear();
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size() > 0){
            for (int i = 0; i < A.size(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++){
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
                idx.clear();
            }
            
        }
        model.update();
        
        vector<int> slackIndices;
        int numSlackVar = 0;
        for (int i = 0; i < c.size(); i++){
            if (c[i] > 0){
                slackIndices.push_back(i);
                numSlackVar++;
            }
        }

        GRBVar bVars[numSlackVar];
        for (int i = 0; i < numSlackVar; i++)
            bVars[i] = model.addVar(0, 1, 0, GRB_BINARY, "y");
        model.update(); 
        
        GRBVar* y = 0;
        y = model.getVars();

        // inequality        
        for (int i = 0; i < numSlackVar; i++ ){
            GRBLinExpr expr = 0;
            expr += 1.0 * x[slackIndices[i]];
            expr -= 100.0 * y[i+numX]; //+numX
            model.addConstr(expr <= 0, "ineq2");
        }
        model.update();   

        // equality
        vector<GRBVar> variables;
        int previousL = 0;
        for (int i = 0; i < numSlackVar; i++ ){
            if (i != 0 && slackIndices[i] - previousL > 2){
                GRBLinExpr expr = 0;
                //expr = grb.LinExpr(ones(len(variables)), variables)
                //model.addConstr(expr, grb.GRB.EQUAL, len(variables) -1)
                for (int j = 0 ; j < variables.size() ; j ++)
                    expr += variables[j];
                model.addConstr(expr == variables.size()-1, "eq2");
                //delete varIndices; int varIndices[MAX]; k =0;
                variables.clear();
                variables.push_back(y[i+numX]);
            }
            else if (slackIndices[i] != 0){
                variables.push_back(y[i+numX]);
            }
            previousL = slackIndices[i];
        }
              
        if (variables.size() > 1){
            GRBLinExpr expr = 0;
            for (int i = 0; i < variables.size(); i++)
                expr += variables[i];
            model.addConstr(expr == variables.size()-1, "last");
        }
        model.update();    
        // model.setObjective(expr,GRB_MINIMIZE);
        model.optimize();
        
        //if (model.get(GRB_IntAttr_Status )==GRB_INFEASIBLE){
            //GRBConstr* con = 0;
            //cout  << "The  model is  infeasible; computing  IIS" << endl;
            //model.computeIIS ();
            //cout  << "\nThe  following  constraint(s) "<< "cannot  be  satisfied:" << endl;
            //con = model.getConstrs ();
            //for(int i = 0; i < model.get(GRB_IntAttr_NumConstrs ); ++i){
                //if(con[i].get(GRB_IntAttr_IISConstr) == 1){
                    //cout  << con[i].get(GRB_StringAttr_ConstrName) << endl;
                //}
            //}
        //}

        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list result;
        result.append(time);
        for (int i = 0; i < c.size(); i++){
            result.append(cVars[i].get(GRB_DoubleAttr_X));
            // cout << extract<double>(result[i]) << endl;
        }
        
        return result;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }
    
    
}

result solveMIP_cost (list& c_, list& A_, list& b_, list& E_, list& e_, int nVarEnd, list& goal_)
{   
    result res;
    // python::list to std::vector 
    vector<double> c;
    massadd(c_, c);
    
    vector< vector<double> > A;
    massadd2(A_, A);
    
    vector<double> b;
    massadd(b_, b);
    
    vector< vector<double> > E;
    massadd2(E_, E);
    
    vector<double> e;
    massadd(e_, e);
    
    vector<double> goal;
    massadd(goal_,goal);
    
    
    try {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        
        int numC = c.size();
        GRBVar cVars[numC];
        
        //add continuous variables
        for (int i = 0 ; i < numC; i++){
            if (i == numC-nVarEnd || i == numC-nVarEnd+1 || i == numC-nVarEnd+2)    // for the cost function
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 1.0, GRB_CONTINUOUS);
            else 
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);
        }
        model.update();
        
        GRBVar* x = 0;
        x = model.getVars();  
        int numX = model.get(GRB_IntAttr_NumVars);        
        
        //// equality constraints
        if (E.size() > 0){
            for (int i = 0; i < E.size(); i ++){
                //idx = [j for j, el in enumerate(E[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++){
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }
                
                //variables = x[idx]
                //coeff = E[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.EQUAL, e[i])
                model.addConstr(expr == e[i]);
                //cout << expr << endl;
                idx.clear();
            }
        }
        model.update();
        
        //// inequality constraints
        if (A.size() > 0){
            for (int i = 0; i < A.size(); i ++){
                //idx = [j for j, el in enumerate(A[i].tolist()) if el != 0.]
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++){
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }
                        
                //variables = x[idx]
                //coeff = a[i,idx]
                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++){
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                //expr = grb.LinExpr(coeff, variables)
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                //model.addConstr(expr, grb.GRB.LESS_EQUAL, b[i])
                model.addConstr(expr <= b[i]);
                //cout << expr << endl;
                idx.clear();
            }
            
        }
        model.update();
        
        vector<int> slackIndices;
        int numSlackVar = 0;
        for (int i = 0; i < numC; i++){
            if (c[i] > 0){
                slackIndices.push_back(i);
                numSlackVar++;
            }
        }

        GRBVar bVars[numSlackVar];
        for (int i = 0; i < numSlackVar; i++)
            bVars[i] = model.addVar(0, 1, 0, GRB_BINARY, "y");
        model.update(); 
        
        GRBVar* y = 0;
        y = model.getVars();

        // inequality        
        for (int i = 0; i < numSlackVar; i++ ){
            GRBLinExpr expr = 0;
            expr += 1.0 * x[slackIndices[i]];
            expr -= 100.0 * y[i+numX]; //+numX
            model.addConstr(expr <= 0, "ineq2");
        }
        model.update();   

        // equality
        vector<GRBVar> variables;
        int previousL = 0;
        for (int i = 0; i < numSlackVar; i++ ){
            if (i != 0 && slackIndices[i] - previousL > 2){
                GRBLinExpr expr = 0;
                //expr = grb.LinExpr(ones(len(variables)), variables)
                //model.addConstr(expr, grb.GRB.EQUAL, len(variables) -1)
                for (int j = 0 ; j < variables.size() ; j ++)
                    expr += variables[j];
                model.addConstr(expr == variables.size()-1, "eq2");
                //delete varIndices; int varIndices[MAX]; k =0;
                variables.clear();
                variables.push_back(y[i+numX]);
            }
            else if (slackIndices[i] != 0){
                variables.push_back(y[i+numX]);
            }
            previousL = slackIndices[i];
        }
              
        if (variables.size() > 1){
            GRBLinExpr expr = 0;
            for (int i = 0; i < variables.size(); i++)
                expr += variables[i];
            model.addConstr(expr == variables.size()-1, "last");
        }
        model.update();    
        
        GRBQuadExpr expr = 0;
        // distance cost function
        GRBLinExpr cx_end_diff = x[numC-nVarEnd]   - goal[0];
        GRBLinExpr cy_end_diff = x[numC-nVarEnd+1] - goal[1];
        GRBLinExpr cz_end_diff = x[numC-nVarEnd+2] - goal[2];
        expr += cx_end_diff * cx_end_diff + cy_end_diff * cy_end_diff + cz_end_diff * cz_end_diff;
        
        model.setObjective(expr,GRB_MINIMIZE);
        model.optimize();

        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        
        list r;
        res.time = time;
        for (int i = 0; i < numC; i++){
            r.append(cVars[i].get(GRB_DoubleAttr_X));
            //cout << extract<double>(result[i]) << endl;
        }
        res.x = r;
        
        return res;
    
    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }
    
}

BOOST_PYTHON_MODULE(qpg) {
    // An established convention for using boost.python.
    using namespace boost::python;

    // Expose the functions
    def("solveLP", solveLP);
    def("solveLP_cost", solveLP_cost);
    def("solveLP_cost_init", solveLP_cost_init);
    def("solveLP_cost_iter", solveLP_cost_iter);
    def("solveMIP", solveMIP);
    def("solveMIP_cost", solveMIP_cost);
    def("solveglpk", solveglpk);
    class_<ResultData>("ResultData")
        .def_readwrite("x", &ResultData::x)
        .def_readwrite("time", &ResultData::time);
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
    
    for (int i = 0; i < 3; i ++)
    {
        list tmp;
        for (int j = 0; j < 3; j++)
            tmp.append(MATRIX_A[i][j]);
        A.append(tmp);
    }
    
    for (int i = 0; i < 3; i++)
        b.append(MATRIX_b[i]);
    
    for (int i = 0; i < 1; i ++)
    {
        list tmp;
        for (int j = 0; j < 3; j++)
            tmp.append(MATRIX_C[i][j]);
        C.append(tmp);
    }
    
    for (int i = 0; i < 1; i++)
        d.append(MATRIX_d[i]);
    
    
    cout << "solveLP" << endl;
    result = solveLP(b, A, b, C, d);
    cout << "solveMIP" << endl;
    result = solveMIP(b, A, b, C, d);
    cout << "solveglpk" << endl;
    result = solveglpk(b, A, b, C, d);

    return 0;
}


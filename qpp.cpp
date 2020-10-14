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
    int status;
    bool success;
    double cost;
    double time;

} resultData;

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

resultData solveglpk (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    resultData result;
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
    int ia[1 + 20000];
    int ja[1 + 20000];
    double ar[1 + 20000];
    lp = glp_create_prob();
    glp_set_prob_name(lp, "lp");
    glp_set_obj_dir(lp, GLP_MIN);

    // ROWS
    const int numEqConstraints = CE.size();
    const int numIneqConstraints = CI.size();
    const int num_constraints_total = numEqConstraints + numIneqConstraints;
    glp_add_rows(lp, num_constraints_total);
    int idrow = 1;
    int idcol = 1;
    int idConsMat = 1;
    int xsize = 0;
    if (CE.size() != 0) xsize = CE[0].size();
    if (CI.size() != 0) xsize = CI[0].size();

    for (int i = 0; i < numIneqConstraints; ++i, ++idrow)
    {
        glp_set_row_bnds(lp, idrow, GLP_UP, 0.0, ci0[i]);
        for (int j = 0; j < xsize; ++j, ++idcol)
        {
            if (CI[i][j] != 0.)
            {
                ia[idConsMat] = idrow, ja[idConsMat] = idcol, ar[idConsMat] = CI[i][j]; // a[1,1] = 1
                ++idConsMat;
            }
        }
        idcol = 1;
    }
    for (int i = 0; i < numEqConstraints; ++i, ++idrow)
    {
        glp_set_row_bnds(lp, idrow, GLP_FX, ce0[i], ce0[i]);
        for (int j = 0; j < xsize; ++j, ++idcol)
        {
            if (CE[i][j] != 0.)
            {
                ia[idConsMat] = idrow, ja[idConsMat] = idcol, ar[idConsMat] = CE[i][j]; // a[1,1] = 1
                ++idConsMat;
            }
        }
        idcol = 1;
    }

    // COLUMNS
    glp_add_cols(lp, xsize);
    for (int i = 0; i < xsize; ++i, ++idcol)
    {
        glp_set_col_bnds(lp, idcol, GLP_FR, UNBOUNDED_DOWN, UNBOUNDED_UP);
        glp_set_obj_coef(lp, idcol, g0[i]);
    }
    glp_load_matrix(lp, idConsMat - 1, ia, ja, ar);

    int res = glp_simplex(lp, &opts);
    const clock_t end_time = clock();
    result.status = glp_get_status(lp);
    double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;

    list r;
    result.time = time;
    if (result.status == GLP_OPT)
    {
        result.success = true;
        cost = glp_get_obj_val(lp);  // obtains a computed value of the objective function
        idrow = 1;
        for (int i = 0; i < xsize; ++i, ++idrow) r.append(glp_get_col_prim(lp, idrow));
        result.cost = glp_get_obj_val(lp);
    }
    else
    {
        result.success = false;
        result.cost = 0;
    }
    result.x = r;

    glp_delete_prob(lp);  // calls the routine glp_delete_prob, which frees all the memory
    glp_free_env();

    // for (int i = 0; i < xsize; ++i)
        // cout << extract<double>(result[i]) << endl;

    return result;
}

resultData solveQP (list& C_, list& c_, list& A_, list& b_, list& E_, list& e_)
{
    resultData result;
    // python::list to std::vector

    vector< vector<double> > C;
    massadd2(C_, C);

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


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        // add real variables
        for (int i = 0 ; i < c.size(); i++)
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);

        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        // equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        //// inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();

        // objective function : QUAD
        GRBQuadExpr obj = 0;
        for (int j = 0; j < c.size(); j++)
            obj += c[j]*x[j];
        for (int i = 0; i < C.size(); i++)
            for (int j = 0; j < C[i].size(); j++)
            if (C[i][j] != 0)
                obj += C[i][j]*x[i]*x[j];
        model.setObjective(obj,GRB_MINIMIZE);
        model.optimize();

        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;

        list r;
        result.time = time;
        result.status = model.get(GRB_IntAttr_Status);
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            cout << cost << endl;
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }

}


resultData solveLP_mindist (list& c_, list& A_, list& b_, list& E_, list& e_, list& goal_, int index)
{
    resultData result;

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
    massadd(goal_, goal);


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        //add continuous variables
        // add real variables
        for (int i = 0 ; i < c.size(); i++)
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);

        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        // equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        //// inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();


        GRBQuadExpr obj = 0;
        // distance to the goal cost
        obj = (goal[0]-x[index])*(goal[0]-x[index])+(goal[1]-x[index+1])*(goal[1]-x[index+1])+(goal[2]-x[index+2])*(goal[2]-x[index+2]);

        model.setObjective(obj,GRB_MINIMIZE);
        model.optimize();
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        result.status = model.get(GRB_IntAttr_Status);

        list r;
        result.time= time;
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        if (model.get(GRB_IntAttr_IsMIP) == 0) {
            throw GRBException("Model is not a MIP");
        }

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }


}


resultData solveLP (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    resultData result;
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


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        // add real variables
        for (int i = 0 ; i < c.size(); i++)
            cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, c[i], GRB_CONTINUOUS);

        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        // equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        //// inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();

        // objective function : slack variables sum
        GRBLinExpr obj = 0;
        for (int i = 0; i <c.size(); i++)
            obj += x[i]*c[i];

        model.setObjective(obj,GRB_MINIMIZE);
        model.optimize();

        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;

        list r;
        result.time = time;
        result.status = model.get(GRB_IntAttr_Status);
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            cout << cost << endl;
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        //cout << e.getMessage() << endl;
    } catch (error_already_set) {
        PyErr_Print();
    }

}

resultData solveMIP (list& c_, list& A_, list& b_, list& E_, list& e_)
{
    resultData result;

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


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        // add continuous variables
        for (int i = 0 ; i < c.size(); i++)
        {
            if (c[i] > 0)    // alpha
                cVars[i] = model.addVar(0, 1, 1, GRB_BINARY, "slack");
            else            // real variables
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x");
        }
        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        // equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        // inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();

        vector<int> slackIndices;
        for (int i = 0; i < c.size(); i++)
        {
            if (c[i] > 0)
                slackIndices.push_back(i);
        }

        // equality slack sum
        vector<GRBVar> variables;
        int previousL = 0;
        for (int i = 0; i < slackIndices.size(); i++ )
        {
            if (i != 0 && slackIndices[i] - previousL > 2)
            {
                GRBLinExpr expr = 0;
                for (int j = 0 ; j < variables.size() ; j ++)
                    expr += variables[j];
                model.addConstr(expr == variables.size()-1, "eq2");
                variables.clear();
                variables.push_back(x[slackIndices[i]]);
            }
            else if (slackIndices[i] != 0)
                variables.push_back(x[slackIndices[i]]);
            previousL = slackIndices[i];
        }

        if (variables.size() > 1)
        {
            GRBLinExpr expr = 0;
            for (int i = 0; i < variables.size(); i++)
                expr += variables[i];
            model.addConstr(expr == variables.size()-1, "last");
        }
        model.update();
        model.optimize();
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        result.status = model.get(GRB_IntAttr_Status);

        list r;
        result.time= time;
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        if (model.get(GRB_IntAttr_IsMIP) == 0) {
            throw GRBException("Model is not a MIP");
        }

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }


}

resultData solveMIP_QP (list& C_, list& c_, list& A_, list& b_, list& E_, list& e_)
{
    resultData result;

    // python::list to std::vector
    vector< vector<double> > C;
    massadd2(C_, C);

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


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        //add continuous variables
        for (int i = 0 ; i < c.size(); i++)
        {
            if (c[i] > 0)    // alpha
                cVars[i] = model.addVar(0, 1, 1, GRB_BINARY, "slack");
            else            // real variables
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x");
        }
        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        //// equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        //// inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();

        vector<int> slackIndices;
        for (int i = 0; i < c.size(); i++)
        {
            if (c[i] > 0)
                slackIndices.push_back(i);
        }

        // equality slack sum
        vector<GRBVar> variables;
        int previousL = 0;
        for (int i = 0; i < slackIndices.size(); i++ )
        {
            if (i != 0 && slackIndices[i] - previousL > 2)
            {
                GRBLinExpr expr = 0;
                for (int j = 0 ; j < variables.size() ; j ++)
                    expr += variables[j];
                model.addConstr(expr == variables.size()-1, "eq2");
                variables.clear();
                variables.push_back(x[slackIndices[i]]);
            }
            else if (slackIndices[i] != 0)
                variables.push_back(x[slackIndices[i]]);
            previousL = slackIndices[i];
        }

        if (variables.size() > 1)
        {
            GRBLinExpr expr = 0;
            for (int i = 0; i < variables.size(); i++)
                expr += variables[i];
            model.addConstr(expr == variables.size()-1, "last");
        }
        model.update();


        // objective function : QUAD
        GRBQuadExpr obj = 0;
        for (int j = 0; j < c.size(); j++)
            obj += c[j]*x[j];
        for (int i = 0; i < C.size(); i++)
            for (int j = 0; j < C[i].size(); j++)
            if (C[i][j] != 0)
                obj += C[i][j]*x[i]*x[j];
        model.setObjective(obj,GRB_MINIMIZE);
        model.optimize();

        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        result.status = model.get(GRB_IntAttr_Status);

        list r;
        result.time= time;
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        if (model.get(GRB_IntAttr_IsMIP) == 0) {
            throw GRBException("Model is not a MIP");
        }

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }


}

resultData solveMIP_mindist (list& c_, list& A_, list& b_, list& E_, list& e_, list& goal_, int index)
{
    resultData result;

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
    massadd(goal_, goal);


    try
    {
        const clock_t begin_time = clock();
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.getEnv().set(GRB_StringParam_LogFile, "");
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);

        GRBVar cVars[c.size()];

        //add continuous variables
        for (int i = 0 ; i < c.size(); i++)
        {
            if (c[i] > 0)    // alpha
                cVars[i] = model.addVar(0, 1, 1, GRB_BINARY, "slack");
            else            // real variables
                cVars[i] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, "x");
        }
        model.update();

        GRBVar* x = 0;
        x = model.getVars();

        //// equality constraints
        if (E.size() > 0)
        {
            for (int i = 0; i < E.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < E[i].size(); j++)
                {
                    if (E[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = E[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr == e[i]);
                idx.clear();
            }
        }
        model.update();

        //// inequality constraints
        if (A.size() > 0)
        {
            for (int i = 0; i < A.size(); i ++)
            {
                vector<int> idx;
                for (int j = 0; j < A[i].size(); j++)
                {
                    if (A[i][j] != 0.0)
                        idx.push_back(j);
                }

                GRBVar variables[idx.size()];
                double coeff[idx.size()];
                for (int j = 0; j < idx.size(); j++)
                {
                    variables[j] = x[idx[j]];
                    coeff[j] = A[i][idx[j]];
                }
                GRBLinExpr expr = 0;
                expr.addTerms(coeff, variables, idx.size());
                model.addConstr(expr <= b[i]);
                idx.clear();
            }

        }
        model.update();

        vector<int> slackIndices;
        for (int i = 0; i < c.size(); i++)
        {
            if (c[i] > 0)
                slackIndices.push_back(i);
        }

        // equality slack sum
        vector<GRBVar> variables;
        int previousL = 0;
        for (int i = 0; i < slackIndices.size(); i++ )
        {
            if (i != 0 && slackIndices[i] - previousL > 2)
            {
                GRBLinExpr expr = 0;
                for (int j = 0 ; j < variables.size() ; j ++)
                    expr += variables[j];
                model.addConstr(expr == variables.size()-1, "eq2");
                variables.clear();
                variables.push_back(x[slackIndices[i]]);
            }
            else if (slackIndices[i] != 0)
                variables.push_back(x[slackIndices[i]]);
            previousL = slackIndices[i];
        }

        if (variables.size() > 1)
        {
            GRBLinExpr expr = 0;
            for (int i = 0; i < variables.size(); i++)
                expr += variables[i];
            model.addConstr(expr == variables.size()-1, "last");
        }
        model.update();


        GRBQuadExpr obj = 0;
        // distance to the goal cost
        obj = (goal[0]-x[index])*(goal[0]-x[index])+(goal[1]-x[index+1])*(goal[1]-x[index+1])+(goal[2]-x[index+2])*(goal[2]-x[index+2]);

        model.setObjective(obj,GRB_MINIMIZE);
        model.optimize();
        const clock_t end_time = clock();
        double time = double(end_time - begin_time)/CLOCKS_PER_SEC*1000;
        result.status = model.get(GRB_IntAttr_Status);

        list r;
        result.time= time;
        double cost = 0;

        if (result.status == 2)
        {
            result.success = true;
            for (int i = 0; i < c.size(); i++)
                r.append(cVars[i].get(GRB_DoubleAttr_X));
            cost = model.get(GRB_DoubleAttr_ObjVal);
            result.cost = cost;
        }
        else
        {
            result.success = false;
        }
        result.x = r;

        if (model.get(GRB_IntAttr_IsMIP) == 0) {
            throw GRBException("Model is not a MIP");
        }

        return result;

    } catch(GRBException e){
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (error_already_set) {
        //PyErr_Print();
    }


}


BOOST_PYTHON_MODULE(qpp) {
    // An established convention for using boost.python.
    using namespace boost::python;

    // Expose the functions
    // def("solveglpk", solveglpk);
    def("solveQP", solveQP);
    def("solveLP_mindist", solveLP_mindist);
    def("solveLP", solveLP);
    def("solveMIP", solveMIP);
    def("solveMIP_QP", solveMIP_QP);
    def("solveMIP_mindist", solveMIP_mindist);
    class_<ResultData>("ResultData")
        .def_readwrite("x", &ResultData::x)
        .def_readwrite("status", &ResultData::status)
        .def_readwrite("success", &ResultData::success)
        .def_readwrite("cost", &ResultData::cost)
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
    resultData result;

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

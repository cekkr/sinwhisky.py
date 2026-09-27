/*
 * flt.c - Fast Linear Transform
 * SinWaver
 *
 * Reconstructed from the printed 2014 source supplied by the user.
 * Original header: "Created by Riccardo Cecchini on 05/09/14."
 *
 * The code below preserves the original data model and elimination algorithm,
 * while fixing obvious syntax/allocation issues so that it is valid C.
 */

#include <stdlib.h>
#include <stddef.h>

typedef struct structFLT FLT;
typedef struct structEq3 Eq3;
typedef struct structVar3 Var3;
typedef struct structValeq3 Valeq3;

struct structFLT {
    int n;
    double *values;

    double real; /* Before */
    double tran; /* After */
};

FLT **generateFltTable(int num)
{
    if (num <= 0)
        return NULL;

    FLT **flts = (FLT **)calloc((size_t)num, sizeof(FLT *));
    if (!flts)
        return NULL;

    /* Inits table */
    for (int i = 0; i < num; ++i) {
        flts[i] = (FLT *)calloc(1, sizeof(FLT));
        if (!flts[i])
            goto fail;

        flts[i]->n = i;
        flts[i]->values = (double *)calloc((size_t)num, sizeof(double));
        if (!flts[i]->values)
            goto fail;

        for (int j = 0; j < num; ++j) {
            if (i == j)
                flts[i]->values[j] = 1.0;
            else
                flts[i]->values[j] = 0.0;
        }

        flts[i]->real = 0.0;
        flts[i]->tran = (double)num;
    }

    return flts;

fail:
    for (int i = 0; i < num; ++i) {
        if (flts[i]) {
            free(flts[i]->values);
            free(flts[i]);
        }
    }
    free(flts);
    return NULL;
}

/* THREE */
struct structEq3 {
    int pos;
    double overadd;
    double *values;
};

struct structVar3 {
    int var;
    Valeq3 **vals;
    double molt;
    double overadd;
    int numvars;
};

struct structValeq3 {
    int var;
    double molt;
};

int fuseEquation2Var(Eq3 *eq, Var3 *var, int length)
{
    int ret = 0;

    if (eq->values[var->var] != 0.0) {
        for (int i = 0; i < length; ++i) {
            if (i != var->var)
                var->vals[i]->molt -= eq->values[i] / eq->values[var->var];
        }

        var->overadd += eq->overadd / eq->values[var->var];
        var->molt++;
        ret = 1;
    }

    return ret;
}

void addOrderedVar3(Var3 **eqs, Var3 *eq, int length)
{
    int i = 0;
    for (; i < length; ++i) {
        if (eq->numvars < eqs[i]->numvars)
            break;
    }

    /* Move */
    for (int j = length; j > i; --j)
        eqs[j] = eqs[j - 1];

    /* Put */
    eqs[i] = eq;
}

void moltVar3(Var3 *var, double molt, int length)
{
    for (int i = 0; i < length; ++i)
        var->vals[i]->molt *= molt;

    var->overadd *= molt;
}

void linkVar3Vars(Var3 **eqs, Var3 *eq, int length, int allength)
{
    int num = eq->var;

    for (int i = 0; i < length; ++i) {
        Var3 *teq = eqs[i];
        int teqvar = teq->var;

        if (teq != eq && num >= 0 && teq->vals[num]->molt != 0.0) {
            /* Correct other vars */
            for (int j = 0; j < allength; ++j) {
                Valeq3 *tval = teq->vals[j];

                if (tval->var != teqvar && eq->vals[j]->molt != 0.0) {
                    double oldmolt = tval->molt;
                    tval->molt += eq->vals[j]->molt * teq->vals[num]->molt;

                    if (tval->molt != 0.0 && oldmolt == 0.0)
                        teq->numvars++;
                }
            }

            /* Correct principals */
            teq->overadd += eq->overadd * teq->vals[num]->molt;

            if (teqvar >= 0 && eq->vals[teqvar]->molt != 0.0) {
                double premolt = eq->vals[teqvar]->molt * teq->vals[num]->molt;

                if (teq->vals[teqvar]->molt != 0.0) {
                    premolt += teq->vals[teqvar]->molt;
                    teq->vals[teqvar]->molt = 0.0;
                }

                /* The printout computes premolt but never uses it. Preserve the
                   original normalization expression while avoiding dead-variable warnings. */
                (void)premolt;
                double denom = 1.0 - (eq->vals[teqvar]->molt * teq->vals[num]->molt);
                if (denom != 0.0) {
                    double tomolt = 1.0 / denom;
                    moltVar3(teq, tomolt, allength);
                }
            }

            if (teq->vals[num]->molt != 0.0) {
                teq->vals[num]->molt = 0.0;
                teq->numvars--;
            }

            /* "Update ts" was present but empty in the printed source. */
        }
    }
}

void flt3(FLT **flts)
{
    if (!flts)
        return;

    int length = (int)flts[0]->tran;
    if (length <= 0)
        return;

    /* Creating equations */
    Eq3 **equations = (Eq3 **)calloc((size_t)length, sizeof(Eq3 *));
    Var3 **varsorder = (Var3 **)calloc((size_t)length + 1U, sizeof(Var3 *));
    Var3 **vars = (Var3 **)calloc((size_t)length, sizeof(Var3 *));
    if (!equations || !varsorder || !vars)
        goto cleanup_outer;

    for (int i = 0; i < length; ++i) {
        equations[i] = (Eq3 *)calloc(1, sizeof(Eq3));
        if (!equations[i])
            goto cleanup;

        equations[i]->pos = i;
        equations[i]->overadd = flts[i]->real;
        equations[i]->values = (double *)calloc((size_t)length, sizeof(double));
        if (!equations[i]->values)
            goto cleanup;

        for (int j = 0; j < length; ++j) {
            double amp = flts[j]->values[i];
            equations[i]->values[j] = amp;
        }
    }

    int vorderlen = 0;

    /* Creating vars */
    for (int i = 0; i < length; ++i) {
        vars[i] = (Var3 *)calloc(1, sizeof(Var3));
        if (!vars[i])
            goto cleanup;

        vars[i]->molt = 0.0;
        vars[i]->overadd = 0.0;
        vars[i]->var = i;
        vars[i]->numvars = 0;

        /* Sets vals */
        vars[i]->vals = (Valeq3 **)calloc((size_t)length, sizeof(Valeq3 *));
        if (!vars[i]->vals)
            goto cleanup;

        for (int j = 0; j < length; ++j) {
            vars[i]->vals[j] = (Valeq3 *)calloc(1, sizeof(Valeq3));
            if (!vars[i]->vals[j])
                goto cleanup;
            vars[i]->vals[j]->var = j;
            vars[i]->vals[j]->molt = 0.0;
        }

        /* From eqs to var */
        for (int j = 0; j < length; ++j)
            fuseEquation2Var(equations[j], vars[i], length);

        /* Check numvars */
        for (int j = 0; j < length; ++j) {
            if (vars[i]->vals[j]->molt != 0.0)
                vars[i]->numvars++;
        }

        /* Divide equations */
        if (vars[i]->molt == 0.0) {
            vars[i]->var = -1; /* This var doesn't exist */
        } else {
            for (int j = 0; j < length; ++j)
                vars[i]->vals[j]->molt /= vars[i]->molt;

            vars[i]->overadd /= vars[i]->molt;
            vars[i]->molt = 1.0;

            /* Add to ordered list */
            addOrderedVar3(varsorder, vars[i], vorderlen);
            vorderlen++;
        }
    }

    /* Solve equations up to down */
    for (int i = 0; i < vorderlen; ++i) {
        Var3 *var = varsorder[i];
        linkVar3Vars(varsorder, var, vorderlen, length);
    }

    /* Get results */
    for (int i = 0; i < length; ++i)
        flts[i]->tran = 0.0;

    for (int i = 0; i < vorderlen; ++i) {
        Var3 *var = varsorder[i];
        if (var && var->var >= 0)
            flts[var->var]->tran = var->overadd;
    }

cleanup:
    if (vars) {
        for (int i = 0; i < length; ++i) {
            if (vars[i]) {
                if (vars[i]->vals) {
                    for (int j = 0; j < length; ++j)
                        free(vars[i]->vals[j]);
                    free(vars[i]->vals);
                }
                free(vars[i]);
            }
        }
    }

    if (equations) {
        for (int i = 0; i < length; ++i) {
            if (equations[i]) {
                free(equations[i]->values);
                free(equations[i]);
            }
        }
    }

cleanup_outer:
    free(varsorder);
    free(vars);
    free(equations);
}

double *confirmFlt3(FLT **flts, int length)
{
    if (!flts || length <= 0)
        return NULL;

    double *ret = (double *)calloc((size_t)length, sizeof(double));
    if (!ret)
        return NULL;

    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < length; ++j)
            ret[j] += flts[i]->tran * flts[i]->values[j];
    }

    return ret;
}

void freeFlts(FLT **flts, int length)
{
    if (!flts)
        return;

    for (int i = 0; i < length; ++i) {
        if (flts[i]) {
            free(flts[i]->values);
            free(flts[i]);
        }
    }

    free(flts);
}

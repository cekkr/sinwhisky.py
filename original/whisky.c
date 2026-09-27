/*
 * whisky.c
 * SinWaver
 *
 * Reconstructed from the printed 2014 source supplied by the user.
 * Original header: "Created by Riccardo Cecchini on 02/09/14."
 *
 * The reconstruction keeps the original algorithm and naming as closely as
 * possible, while fixing obvious C syntax issues and a few unsafe mechanical
 * transcription problems so this file can be compiled as C.
 */

#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct structCircle {
    float xs[3]; /* Three points x-axis */
    float ys[3]; /* Three points y-axis */

    char isCircle;
    float oX, oY, r; /* Center and radius */
    float apr;       /* Aperture */

    float m;         /* m, if it's a line */

    /* Results */
    int zooming;
    int numRes;
    float *zoomRes;
} Circle;

static float maxf2(float a, float b)
{
    return (a > b) ? a : b;
}

float getYinLine(Circle *circle, float x)
{
    circle->m = (circle->ys[0] - circle->ys[2]) /
                (circle->xs[0] - circle->xs[2]);
    return circle->m * (x - circle->xs[0]);
}

float fround(float f, int precision)
{
    float totPrec = powf(10.0f, (float)precision);
    return roundf(f * totPrec) / totPrec;
}

/* Radiants time! */
float diffRadiants(float t1, float t2, int verse)
{
    float tr = 0.0f;

    if (verse > 0) {
        if (t2 < t1)
            t2 += (float)M_PI * 2.0f;
        tr = t2 - t1;
    } else {
        if (t2 > t1)
            t2 = -(((float)M_PI * 2.0f) - t2);
        tr = t1 - t2;
    }

    return fabsf(tr);
}

/* Relative calculations */
float getRelXCircle(float x, Circle *circle)
{
    return (x - circle->oX) / circle->r;
}

float getAbsXCircle(float x, Circle *circle)
{
    return (x * circle->r) + circle->oX;
}

float getRelYCircle(float y, Circle *circle)
{
    return (y - circle->oY) / circle->r;
}

float getAbsYCircle(float y, Circle *circle)
{
    return (y * circle->r) + circle->oY;
}

/* Calc t in a circle */
float getFloatInLimits(float r)
{
    if (r > 1.0f)
        r = 1.0f;
    if (r < -1.0f)
        r = -1.0f;
    return r;
}

double getTfromY(float x, float y, Circle *circle)
{
    x = getFloatInLimits(getRelXCircle(x, circle));
    y = getFloatInLimits(getRelYCircle(y, circle));

    double t = acos((double)x);

    if (y < 0.0f) {
        if (t >= 0.0)
            t = M_PI + (M_PI - t);
        else
            t = (M_PI * 2.0) - t;
    }

    return t;
}

/*
 * Original source referenced:
 * http://mathforum.org/library/drmath/view/54323.html
 */
void findCircle(Circle *circle)
{
    float bx = circle->xs[0];
    float by = circle->ys[0];
    float cx = circle->xs[1];
    float cy = circle->ys[1];
    float dx = circle->xs[2];
    float dy = circle->ys[2];
    float temp = cx * cx + cy * cy;
    float bc = (bx * bx + by * by - temp) / 2.0f;
    float cd = (temp - dx * dx - dy * dy) / 2.0f;

    float det = (bx - cx) * (cy - dy) - (cx - dx) * (by - cy);
    if (fabsf(det) < 1.0e-6f) {
        circle->oX = circle->oY = 1.0f;
        circle->r = 0.0f;
        return;
    }

    det = 1.0f / det;
    circle->oX = (bc * (cy - dy) - cd * (by - cy)) * det;
    circle->oY = ((bx - cx) * cd - (cx - dx) * bc) * det;

    circle->r = sqrtf(powf(circle->oX - bx, 2.0f) +
                      powf(circle->oY - by, 2.0f));
}

int getCircleVerse(float tp[])
{
    int verse;
    if (tp[1] > tp[0])
        verse = 1;
    else
        verse = -1;

    return verse;
}

void processCircle(Circle *circle)
{
    /* Calc aperture */
    float maxapr = 0.0f;
    maxapr = maxf2(fabsf(circle->ys[0] - circle->ys[1]), maxapr);
    maxapr = maxf2(fabsf(circle->ys[1] - circle->ys[2]), maxapr);
    maxapr = maxf2(fabsf(circle->ys[0] - circle->ys[2]), maxapr);

    circle->apr = maxapr;
    for (int i = 0; i < 3; ++i)
        circle->xs[i] *= circle->apr; /* Set the new aperture */

    float minSpace = (circle->xs[2] - circle->xs[0]) /
                     (float)(circle->numRes - 1);

    if (getYinLine(circle, circle->xs[1]) == circle->ys[1]) { /* It's a line! */
        for (int i = 0; i < circle->numRes; ++i) {
            float xNow = circle->xs[0] + (minSpace * (float)i);
            circle->zoomRes[i] = getYinLine(circle, xNow);
        }
    } else { /* It's a circle! */
        findCircle(circle);

        float tp[3];
        for (int i = 0; i < 3; ++i)
            tp[i] = (float)getTfromY(circle->xs[i], circle->ys[i], circle);

        int verse = getCircleVerse(tp);
        float *ts = (float *)malloc((size_t)circle->numRes * sizeof(float));
        if (!ts)
            return;

        /* Calc radiants */
        for (int i = 0; i < 3; ++i) {
            ts[(circle->zooming + 1) * i] = tp[i];

            if (i < 2) {
                float diff = diffRadiants(tp[i], tp[i + 1], verse);
                float inc = diff / (float)(circle->zooming + 1);

                float tnow = tp[i];
                for (int j = 0; j < circle->zooming; ++j) {
                    if (verse > 0)
                        tnow += inc;
                    else
                        tnow -= inc;

                    ts[i * (circle->zooming + 1) + (j + 1)] = tnow;
                }
            }
        }

        /* Calc results */
        for (int i = 0; i < circle->numRes; ++i) {
            float mts = ts[i];
            float y = getAbsYCircle(sinf(mts), circle);
            circle->zoomRes[i] = y;
        }

        free(ts);
    }
}

void shiftFloats(float *floats, int shift, int length)
{
    if (!floats || shift <= 0 || length <= 0)
        return;

    if (shift >= length) {
        for (int i = 0; i < length; ++i)
            floats[i] = 0.0f;
        return;
    }

    /* The printout's first loop was truncated/unsafe; this is the intended shift. */
    for (int i = 0; i < length - shift; ++i)
        floats[i] = floats[i + shift];
    for (int i = length - shift; i < length; ++i)
        floats[i] = 0.0f;
}

float *simpleWhisky(float *samples, int samplesLength, int zooming)
{
    if (!samples || samplesLength <= 0 || zooming < 0)
        return NULL;

    int newLength = (samplesLength - 1) * (1 + zooming) + 1;
    float *samret = (float *)calloc((size_t)newLength, sizeof(float));
    if (!samret)
        return NULL;

    /* Calcolo delle divisioni */
    int zoomedSample = zooming + 1;
    int bufferZoom = (2 * zooming) + 3;
    float *divret = (float *)calloc((size_t)bufferZoom, sizeof(float));
    float *shoret = (float *)calloc((size_t)bufferZoom, sizeof(float));
    if (!divret || !shoret) {
        free(divret);
        free(shoret);
        free(samret);
        return NULL;
    }

    /* Go, go, go! */
    for (int i = 0; i < samplesLength; ++i) {
        if (i > 0 && i < samplesLength - 1) {
            Circle *circle = (Circle *)calloc(1, sizeof(Circle));
            if (!circle)
                break;

            /* Init new circle */
            for (int j = 0; j < 3; ++j)
                circle->xs[j] = (float)j;
            for (int j = 0; j < 3; ++j)
                circle->ys[j] = samples[i + j - 1];

            circle->zooming = zooming;
            circle->numRes = (2 * zooming) + 3;
            circle->zoomRes = (float *)calloc((size_t)circle->numRes, sizeof(float));
            if (!circle->zoomRes) {
                free(circle);
                break;
            }

            /* Process circle */
            processCircle(circle);

            /* Write circle */
            for (int j = 0; j < bufferZoom; ++j) {
                float todiv = sinf(((float)(j + 1) / (float)(bufferZoom + 1)) * (float)M_PI);
                divret[j] += todiv;
                shoret[j] += circle->zoomRes[j] * todiv;
            }

            /* Save the last */
            for (int j = 0; j < (zooming + 1); ++j) {
                if (divret[j] != 0.0f)
                    samret[j + ((i - 1) * zoomedSample)] = shoret[j] / divret[j];
            }

            if (i == samplesLength - 2) { /* Is the last circle (y) */
                for (int j = 1; j <= (zoomedSample + 1); ++j) {
                    int idx = bufferZoom - j;
                    float dis = (divret[idx] != 0.0f) ? shoret[idx] / divret[idx] : 0.0f;
                    samret[newLength - j] = dis;
                }
            }

            free(circle->zoomRes);
            free(circle);

            shiftFloats(divret, zoomedSample, bufferZoom);
            shiftFloats(shoret, zoomedSample, bufferZoom);
        }
    }

    /* We are sure... */
    for (int i = 0; i < samplesLength; ++i)
        samret[i * zoomedSample] = samples[i];

    free(divret);
    free(shoret);

    /* Finish. Maybe. */
    return samret;
}

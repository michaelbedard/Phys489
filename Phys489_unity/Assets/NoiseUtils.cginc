// NoiseUtils.cginc
// Shared noise primitives included by FIFGenerator.compute.

#ifndef NOISE_UTILS_INCLUDED
#define NOISE_UTILS_INCLUDED

static const float PI  = 3.14159265359;
static const float LN2 = 0.693147;

// --- PCG-3D HASH ---
// Uniform quality for any integer-valued input, including large coordinates.
// Ref: "Hash Functions for GPU Rendering", Jarzynski & Olano (2020).
float hash(float3 p)
{
    uint3 q = uint3(int3(p));
    q = q * 1664525u + 1013904223u;
    q.x ^= q.y * q.z;
    q.y ^= q.z * q.x;
    q.z ^= q.x * q.y;
    q ^= q >> 16u;
    return float(q.x ^ q.y ^ q.z) * (1.0 / 4294967296.0);
}

// --- CMS ALPHA-STABLE GENERATOR (Chambers-Mallows-Stuck 1976) ---
// Generates a symmetric alpha-stable random variable with unit scale.
// V ~ Uniform(-pi/2, pi/2),  W ~ Exponential(1)
// X = sin(alpha*V) / cos(V)^(1/alpha) * (cos((1-alpha)*V) / W)^((1-alpha)/alpha)
// Ref: Chambers, Mallows & Stuck (1976), JASA 71(354):340-344.
float levyStable(float3 id, float um_alpha)
{
    float u1 = clamp(hash(id),                          1e-4, 1.0 - 1e-4);
    float u2 = clamp(hash(id + float3(1.0, 1.0, 1.0)), 1e-4, 1.0 - 1e-4);
    float V      = PI * (u1 - 0.5);
    float W      = -log(u2);
    float invA   = 1.0 / um_alpha;
    float omA    = 1.0 - um_alpha;
    return sin(um_alpha * V) * pow(cos(V), -invA)
         * pow(cos(omA * V) / W, omA * invA);
}

#endif // NOISE_UTILS_INCLUDED

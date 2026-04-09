using UnityEngine;

public class SimulationController : MonoBehaviour
{
    [Header("Simulation Settings")]
    public ComputeShader cascadeShader;
    public ComputeShader terrainShader;
    public ComputeShader fifGenerator;
    public int textureResolution = 1024;

    [Header("Visuals (IR Enhancement)")]
    public Gradient irColorMap;
    public Gradient terrainColorMap;
    private Texture2D _gradientTexture;
    private Texture2D _terrainGradientTexture;

    [Header("Cloud UM Parameters")]
    [Range(0, 1)]      public float C1    = 0.1f;
    [Range(0, 1)]      public float H     = 0.33f;
    [Range(1.01f, 2f)] public float alpha = 1.79f;

    [Header("Albedo UM Parameters")]
    [Range(0, 1)]      public float C1_albedo    = 0.12f;
    [Range(0, 1)]      public float H_albedo     = 0.66f;
    [Range(1.01f, 2f)] public float alpha_albedo = 1.79f;

    [Header("Cloud Animation")]
    [Tooltip("UV-space advection velocity (cycles per second). Simulates zonal/meridional wind.")]
    public Vector2 windVelocity = new Vector2(0.0002f, 0.0001f);

    [Header("Space-Time Anisotropy (OU Evolution)")]
    [Tooltip("Enable scale-dependent temporal evolution of cloud Fourier modes (space-time anisotropy). Disable to revert to frozen turbulence.")]
    public bool enableEvolution = true;
    [Tooltip("Dynamical scaling exponent z: tau(k) = tau0 * |k|^(-z). Larger z => small scales evolve faster relative to large scales. Typical atmosphere: ~0.5.")]
    [Range(0f, 2f)] public float z = 0.5f;
    [Tooltip("Base correlation time (seconds) for the largest spatial mode (|k|=1). Controls overall evolution speed.")]
    [Range(0.1f, 60f)] public float tau0 = 5.0f;

    [Header("Preview Toggles")]
    public bool showAlbedo = true;
    public bool showClouds = true;

    // Cloud pipeline
    public RenderTexture RenderTexture { get; private set; }
    public ComputeBuffer PhysicsBuffer { get; private set; }
    private Material     _planetMaterial;
    private int          _cloudKernel;

    // Terrain pipeline
    private RenderTexture _albedoRT;
    private RenderTexture _blackRT;
    private int           _terrainKernel;

    // FIF generator kernels
    private int _genSpecKernel;
    private int _evolveSpecKernel;
    private int _phaseShiftKernel;
    private int _ifftRowKernel;
    private int _ifftColKernel;
    private int _extractKernel;
    private int _frameCount;

    // FIF buffers and textures
    private ComputeBuffer _cloudBaseSpectrumBuffer;   // persistent base Fourier coefficients
    private ComputeBuffer _cloudSpectrumBuffer;        // working buffer (IFFT in-place)
    private ComputeBuffer _albedoBaseSpectrumBuffer;
    private ComputeBuffer _albedoSpectrumBuffer;
    private RenderTexture _cloudLogFieldRT;            // exp(log_field) for cloud
    private RenderTexture _albedoLogFieldRT;           // exp(log_field) for albedo

    // Dirty flags — re-dispatch when UM parameters change in the inspector.
    private float _cachedC1Cloud    = -1f;
    private float _cachedHCloud     = -1f;
    private float _cachedAlphaCloud = -1f;
    private float _cachedC1Albedo    = -1f;
    private float _cachedHAlbedo     = -1f;
    private float _cachedAlphaAlbedo = -1f;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    void Start()
    {
        InitTextures();
        InitGradients();
        InitShaders();
        InitFIFBuffers();
        RegenerateCloudBase();
        DispatchAlbedoFIF();
    }

    void InitTextures()
    {
        RenderTexture = CreateRT(RenderTextureFormat.ARGB32);
        _albedoRT     = CreateRT(RenderTextureFormat.ARGB32);
        _blackRT      = new RenderTexture(1, 1, 0); _blackRT.enableRandomWrite = true; _blackRT.Create();

        _cloudLogFieldRT  = CreateRT(RenderTextureFormat.RFloat);
        _albedoLogFieldRT = CreateRT(RenderTextureFormat.RFloat);

        PhysicsBuffer = new ComputeBuffer(textureResolution * textureResolution, sizeof(float));

        _planetMaterial = GetComponent<Renderer>().material;
        _planetMaterial.mainTexture = RenderTexture;
    }

    RenderTexture CreateRT(RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(textureResolution, textureResolution, 0, fmt);
        rt.enableRandomWrite = true;
        rt.Create();
        return rt;
    }

    void InitGradients()
    {
        _gradientTexture        = BakeGradient(irColorMap,      _gradientTexture);
        _terrainGradientTexture = BakeGradient(terrainColorMap, _terrainGradientTexture);
    }

    Texture2D BakeGradient(Gradient gradient, Texture2D existing)
    {
        if (existing == null)
        {
            existing = new Texture2D(256, 1, TextureFormat.RGBA32, false);
            existing.wrapMode = TextureWrapMode.Clamp;
        }
        for (int i = 0; i < 256; i++)
            existing.SetPixel(i, 0, gradient.Evaluate(i / 255f));
        existing.Apply();
        return existing;
    }

    void InitShaders()
    {
        _cloudKernel   = cascadeShader.FindKernel("CSMain");
        _terrainKernel = terrainShader.FindKernel("CSMain");

        _genSpecKernel    = fifGenerator.FindKernel("GenerateSpectrum");
        _evolveSpecKernel = fifGenerator.FindKernel("EvolveSpectrum");
        _phaseShiftKernel = fifGenerator.FindKernel("PhaseShift");
        _ifftRowKernel    = fifGenerator.FindKernel("IFFTRow");
        _ifftColKernel    = fifGenerator.FindKernel("IFFTCol");
        _extractKernel    = fifGenerator.FindKernel("ExtractField");
    }

    void InitFIFBuffers()
    {
        int count  = textureResolution * textureResolution;
        int stride = 2 * sizeof(float); // float2

        _cloudBaseSpectrumBuffer  = new ComputeBuffer(count, stride);
        _cloudSpectrumBuffer      = new ComputeBuffer(count, stride);
        _albedoBaseSpectrumBuffer = new ComputeBuffer(count, stride);
        _albedoSpectrumBuffer     = new ComputeBuffer(count, stride);
    }

    // -------------------------------------------------------------------------
    // FIF dispatch helpers
    // -------------------------------------------------------------------------

    /// <summary>Generates a fresh cloud base spectrum with a random seed.</summary>
    public void RegenerateCloudBase()
    {
        fifGenerator.SetFloat("C1",    C1);
        fifGenerator.SetFloat("H",     H);
        fifGenerator.SetFloat("Alpha", alpha);
        fifGenerator.SetInt("Seed",    Random.Range(0, 1000000));
        fifGenerator.SetBuffer(_genSpecKernel, "BaseSpectrumBuffer", _cloudBaseSpectrumBuffer);
        fifGenerator.Dispatch(_genSpecKernel, textureResolution / 8, textureResolution / 8, 1);

        _cachedC1Cloud    = C1;
        _cachedHCloud     = H;
        _cachedAlphaCloud = alpha;
    }

    /// <summary>Per-frame cloud dispatch: phase-shift the base, IFFT, extract.</summary>
    void DispatchCloudFrame()
    {
        int groups8 = textureResolution / 8;

        // OU evolution: update BaseSpectrumBuffer in-place with scale-dependent temporal decorrelation.
        // Must run before PhaseShift so advection is applied to the already-evolved spectrum.
        if (enableEvolution)
        {
            fifGenerator.SetFloat("Dt",         Time.deltaTime);
            fifGenerator.SetFloat("Tau0",       tau0);
            fifGenerator.SetFloat("Gz",         z);
            fifGenerator.SetInt("FrameCount",   _frameCount);
            // Re-set cloud UM params explicitly: albedo dispatch earlier in Update() may have overwritten them.
            fifGenerator.SetFloat("C1",         C1);
            fifGenerator.SetFloat("H",          H);
            fifGenerator.SetFloat("Alpha",      alpha);
            fifGenerator.SetBuffer(_evolveSpecKernel, "BaseSpectrumBuffer", _cloudBaseSpectrumBuffer);
            fifGenerator.Dispatch(_evolveSpecKernel, groups8, groups8, 1);
        }
        _frameCount++;

        // Phase-shift (copies base -> working with advection phase)
        fifGenerator.SetFloat("Time", Time.time);
        fifGenerator.SetVector("WindVelocity", new Vector4(windVelocity.x, windVelocity.y, 0, 0));
        fifGenerator.SetBuffer(_phaseShiftKernel, "BaseSpectrumBuffer", _cloudBaseSpectrumBuffer);
        fifGenerator.SetBuffer(_phaseShiftKernel, "SpectrumBuffer",     _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_phaseShiftKernel, groups8, groups8, 1);

        // IFFT
        fifGenerator.SetBuffer(_ifftRowKernel, "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_ifftRowKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_ifftColKernel, "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_ifftColKernel, textureResolution, 1, 1);

        // Extract: exp(log_field) -> LogFieldTex + PhysicsData
        fifGenerator.SetBuffer(_extractKernel,  "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.SetTexture(_extractKernel, "LogFieldTex",    _cloudLogFieldRT);
        fifGenerator.SetBuffer(_extractKernel,  "PhysicsData",    PhysicsBuffer);
        fifGenerator.Dispatch(_extractKernel, groups8, groups8, 1);
    }

    /// <summary>
    /// Generates a fresh albedo FIF realization (new seed), IFFTs it into
    /// _albedoLogFieldRT, then dispatches the terrain visualization shader
    /// to write the coloured result to _albedoRT.
    /// </summary>
    void DispatchAlbedoFIF()
    {
        int groups8 = textureResolution / 8;

        // Generate base spectrum
        fifGenerator.SetFloat("C1",    C1_albedo);
        fifGenerator.SetFloat("H",     H_albedo);
        fifGenerator.SetFloat("Alpha", alpha_albedo);
        fifGenerator.SetInt("Seed",    Random.Range(0, 1000000));
        fifGenerator.SetBuffer(_genSpecKernel, "BaseSpectrumBuffer", _albedoBaseSpectrumBuffer);
        fifGenerator.Dispatch(_genSpecKernel, groups8, groups8, 1);

        // Copy to working buffer (PhaseShift with t=0 = plain copy)
        fifGenerator.SetFloat("Time", 0f);
        fifGenerator.SetVector("WindVelocity", Vector4.zero);
        fifGenerator.SetBuffer(_phaseShiftKernel, "BaseSpectrumBuffer", _albedoBaseSpectrumBuffer);
        fifGenerator.SetBuffer(_phaseShiftKernel, "SpectrumBuffer",     _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_phaseShiftKernel, groups8, groups8, 1);

        // IFFT
        fifGenerator.SetBuffer(_ifftRowKernel, "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_ifftRowKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_ifftColKernel, "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_ifftColKernel, textureResolution, 1, 1);

        // Extract
        fifGenerator.SetBuffer(_extractKernel,  "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.SetTexture(_extractKernel, "LogFieldTex",    _albedoLogFieldRT);
        fifGenerator.SetBuffer(_extractKernel,  "PhysicsData",    PhysicsBuffer);
        fifGenerator.Dispatch(_extractKernel, groups8, groups8, 1);

        // Visualize: apply terrain colour LUT, write to _albedoRT
        DispatchTerrainVisualize();

        _cachedC1Albedo    = C1_albedo;
        _cachedHAlbedo     = H_albedo;
        _cachedAlphaAlbedo = alpha_albedo;
    }

    void DispatchTerrainVisualize()
    {
        _terrainGradientTexture = BakeGradient(terrainColorMap, _terrainGradientTexture);
        terrainShader.SetTexture(_terrainKernel, "AlbedoFIFFieldTex", _albedoLogFieldRT);
        terrainShader.SetTexture(_terrainKernel, "GradientTex",       _terrainGradientTexture);
        terrainShader.SetTexture(_terrainKernel, "Result",            _albedoRT);
        terrainShader.Dispatch(_terrainKernel, textureResolution / 8, textureResolution / 8, 1);
    }

    // -------------------------------------------------------------------------
    // Per-frame update
    // -------------------------------------------------------------------------

    void Update()
    {
        _gradientTexture        = BakeGradient(irColorMap,      _gradientTexture);
        _terrainGradientTexture = BakeGradient(terrainColorMap, _terrainGradientTexture);

        // Re-generate albedo when its UM params change
        if (!Mathf.Approximately(C1_albedo,    _cachedC1Albedo)    ||
            !Mathf.Approximately(H_albedo,     _cachedHAlbedo)     ||
            !Mathf.Approximately(alpha_albedo, _cachedAlphaAlbedo))
        {
            DispatchAlbedoFIF();
        }

        // Re-generate cloud base when its UM params change
        if (!Mathf.Approximately(C1,    _cachedC1Cloud)    ||
            !Mathf.Approximately(H,     _cachedHCloud)     ||
            !Mathf.Approximately(alpha, _cachedAlphaCloud))
        {
            RegenerateCloudBase();
        }

        // Preview toggle: show albedo only
        if (!showClouds && showAlbedo)
        {
            _planetMaterial.mainTexture = _albedoRT;
            return;
        }

        _planetMaterial.mainTexture = RenderTexture;

        // Compute cloud field for this frame
        DispatchCloudFrame();

        // Composite cloud over albedo into final display texture
        cascadeShader.SetTexture(_cloudKernel, "FIFFieldTex", _cloudLogFieldRT);
        cascadeShader.SetTexture(_cloudKernel, "GradientTex", _gradientTexture);
        cascadeShader.SetTexture(_cloudKernel, "AlbedoTex",   showAlbedo ? _albedoRT : _blackRT);
        cascadeShader.SetTexture(_cloudKernel, "Result",      RenderTexture);
        cascadeShader.Dispatch(_cloudKernel, textureResolution / 8, textureResolution / 8, 1);
    }

    // -------------------------------------------------------------------------
    // Snapshot capture (used by DataExporter)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Captures one independent cloud realization by regenerating the base
    /// spectrum with a fresh seed, IFFTing at t=0, and returning the raw field.
    /// </summary>
    public float[] CaptureSnapshot()
    {
        int groups8 = textureResolution / 8;

        // Fresh realization — new random seed
        fifGenerator.SetFloat("C1",    C1);
        fifGenerator.SetFloat("H",     H);
        fifGenerator.SetFloat("Alpha", alpha);
        fifGenerator.SetInt("Seed",    Random.Range(0, 1000000));
        fifGenerator.SetBuffer(_genSpecKernel, "BaseSpectrumBuffer", _cloudBaseSpectrumBuffer);
        fifGenerator.Dispatch(_genSpecKernel, groups8, groups8, 1);

        // Copy (t=0, no advection shift)
        fifGenerator.SetFloat("Time", 0f);
        fifGenerator.SetVector("WindVelocity", Vector4.zero);
        fifGenerator.SetBuffer(_phaseShiftKernel, "BaseSpectrumBuffer", _cloudBaseSpectrumBuffer);
        fifGenerator.SetBuffer(_phaseShiftKernel, "SpectrumBuffer",     _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_phaseShiftKernel, groups8, groups8, 1);

        // IFFT + extract
        fifGenerator.SetBuffer(_ifftRowKernel, "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_ifftRowKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_ifftColKernel, "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.Dispatch(_ifftColKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_extractKernel,  "SpectrumBuffer", _cloudSpectrumBuffer);
        fifGenerator.SetTexture(_extractKernel, "LogFieldTex",    _cloudLogFieldRT);
        fifGenerator.SetBuffer(_extractKernel,  "PhysicsData",    PhysicsBuffer);
        fifGenerator.Dispatch(_extractKernel, groups8, groups8, 1);

        float[] data = new float[textureResolution * textureResolution];
        PhysicsBuffer.GetData(data);
        return data;
    }

    /// <summary>
    /// Captures one independent albedo realization with a fresh seed.
    /// Also refreshes the visual _albedoRT and _albedoLogFieldRT.
    /// </summary>
    public float[] CaptureAlbedoSnapshot()
    {
        int groups8 = textureResolution / 8;

        // Fresh realization
        fifGenerator.SetFloat("C1",    C1_albedo);
        fifGenerator.SetFloat("H",     H_albedo);
        fifGenerator.SetFloat("Alpha", alpha_albedo);
        fifGenerator.SetInt("Seed",    Random.Range(0, 1000000));
        fifGenerator.SetBuffer(_genSpecKernel, "BaseSpectrumBuffer", _albedoBaseSpectrumBuffer);
        fifGenerator.Dispatch(_genSpecKernel, groups8, groups8, 1);

        // Copy (t=0)
        fifGenerator.SetFloat("Time", 0f);
        fifGenerator.SetVector("WindVelocity", Vector4.zero);
        fifGenerator.SetBuffer(_phaseShiftKernel, "BaseSpectrumBuffer", _albedoBaseSpectrumBuffer);
        fifGenerator.SetBuffer(_phaseShiftKernel, "SpectrumBuffer",     _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_phaseShiftKernel, groups8, groups8, 1);

        // IFFT + extract
        fifGenerator.SetBuffer(_ifftRowKernel, "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_ifftRowKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_ifftColKernel, "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.Dispatch(_ifftColKernel, textureResolution, 1, 1);

        fifGenerator.SetBuffer(_extractKernel,  "SpectrumBuffer", _albedoSpectrumBuffer);
        fifGenerator.SetTexture(_extractKernel, "LogFieldTex",    _albedoLogFieldRT);
        fifGenerator.SetBuffer(_extractKernel,  "PhysicsData",    PhysicsBuffer);
        fifGenerator.Dispatch(_extractKernel, groups8, groups8, 1);

        float[] data = new float[textureResolution * textureResolution];
        PhysicsBuffer.GetData(data);

        // Refresh visual albedo after export
        DispatchTerrainVisualize();
        return data;
    }

    /// <summary>
    /// Reads the current cloud field from PhysicsBuffer (written by the last ExtractField dispatch).
    /// No GPU re-dispatch — safe to call at end-of-frame after DispatchCloudFrame has run.
    /// </summary>
    public float[] ReadCurrentField()
    {
        float[] data = new float[textureResolution * textureResolution];
        PhysicsBuffer.GetData(data);
        return data;
    }

    void OnDestroy()
    {
        if (RenderTexture          != null) RenderTexture.Release();
        if (_albedoRT              != null) _albedoRT.Release();
        if (_blackRT               != null) _blackRT.Release();
        if (_cloudLogFieldRT       != null) _cloudLogFieldRT.Release();
        if (_albedoLogFieldRT      != null) _albedoLogFieldRT.Release();
        if (PhysicsBuffer          != null) PhysicsBuffer.Release();
        if (_cloudBaseSpectrumBuffer  != null) _cloudBaseSpectrumBuffer.Release();
        if (_cloudSpectrumBuffer      != null) _cloudSpectrumBuffer.Release();
        if (_albedoBaseSpectrumBuffer != null) _albedoBaseSpectrumBuffer.Release();
        if (_albedoSpectrumBuffer     != null) _albedoSpectrumBuffer.Release();
    }
}

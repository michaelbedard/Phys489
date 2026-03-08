using UnityEngine;

public class SimulationController : MonoBehaviour
{
    [Header("Simulation Settings")]
    public ComputeShader cascadeShader;
    public int textureResolution = 1024;

    [Header("Visuals (IR Enhancement)")]
    public Texture2D earthTexture; 
    public Gradient irColorMap;
    private Texture2D _gradientTexture;

    [Header("UM Parameters")]
    [Range(0, 1)] public float C1 = 1.0f;
    [Range(0, 1)] public float H = 0.3f;

    [Header("Cascade Initialization")]
    [Tooltip("Z-offset into the 3D noise space at t=0. Avoids the degenerate z=0 slice. Physically neutral — the noise field is statistically homogeneous.")]
    public float initialZOffset = 5.0f;

    public RenderTexture RenderTexture { get; private set; }
    public ComputeBuffer PhysicsBuffer { get; private set; }
    private Material _planetMaterial;
    private int _kernelHandle;

    void Start()
    {
        InitTexture();
        InitGradient(); 
        InitShader();
    }

    void InitTexture()
    {
        RenderTexture = new RenderTexture(textureResolution, textureResolution, 0);
        RenderTexture.enableRandomWrite = true;
        RenderTexture.Create();

        PhysicsBuffer = new ComputeBuffer(textureResolution * textureResolution, sizeof(float));

        _planetMaterial = GetComponent<Renderer>().material;
        _planetMaterial.mainTexture = RenderTexture;
    }

    void InitGradient()
    {
        if (_gradientTexture == null)
            _gradientTexture = new Texture2D(256, 1, TextureFormat.RGBA32, false);
        
        _gradientTexture.wrapMode = TextureWrapMode.Clamp; 

        for (int i = 0; i < 256; i++)
        {
            float t = i / 255f;
            _gradientTexture.SetPixel(i, 0, irColorMap.Evaluate(t));
        }
        _gradientTexture.Apply();
    }

    void InitShader()
    {
        _kernelHandle = cascadeShader.FindKernel("CSMain");
    }

    void Update()
    {
        // Useful for debugging colors in Play Mode, disable for release
        InitGradient(); 

        cascadeShader.SetFloat("Time", Time.time);
        cascadeShader.SetFloat("InitialZOffset", initialZOffset);
        cascadeShader.SetFloat("C1", C1);
        cascadeShader.SetFloat("H", H);
        
        // Pass the textures
        cascadeShader.SetTexture(_kernelHandle, "GradientTex", _gradientTexture);
        cascadeShader.SetTexture(_kernelHandle, "Result", RenderTexture);
        cascadeShader.SetBuffer(_kernelHandle, "PhysicsData", PhysicsBuffer);
        
        // If earthTexture is assigned, use it. Otherwise, use a default black texture.
        Texture texToPass = earthTexture != null ? earthTexture : Texture2D.blackTexture;
        cascadeShader.SetTexture(_kernelHandle, "EarthTex", texToPass);

        cascadeShader.SetVector("NoiseOrigin", Vector3.zero);

        int threadGroups = textureResolution / 8;
        cascadeShader.Dispatch(_kernelHandle, threadGroups, threadGroups, 1);
    }

    // Dispatches the cascade at a given noise origin and returns the raw field data.
    // Used by DataExporter to capture independent realizations without waiting.
    public float[] CaptureSnapshot(Vector3 noiseOrigin)
    {
        cascadeShader.SetVector("NoiseOrigin", noiseOrigin);
        cascadeShader.Dispatch(_kernelHandle, textureResolution / 8, textureResolution / 8, 1);

        float[] data = new float[textureResolution * textureResolution];
        PhysicsBuffer.GetData(data); // synchronizing call — waits for GPU to finish
        return data;
    }

    void OnDestroy()
    {
        if (RenderTexture != null) RenderTexture.Release();
        if (PhysicsBuffer != null) PhysicsBuffer.Release();
    }
}
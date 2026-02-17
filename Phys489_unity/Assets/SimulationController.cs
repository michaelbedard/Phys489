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

    public RenderTexture RenderTexture { get; private set; }
    private Material _planetMaterial;
    private int _kernelHandle;

    void Start()
    {
        SetupDefaultThermalGradient();
        
        InitTexture();
        InitGradient(); 
        InitShader();
    }

    void InitTexture()
    {
        RenderTexture = new RenderTexture(textureResolution, textureResolution, 0);
        RenderTexture.enableRandomWrite = true;
        RenderTexture.Create();

        _planetMaterial = GetComponent<Renderer>().material;
        _planetMaterial.mainTexture = RenderTexture;
    }

    void SetupDefaultThermalGradient()
    {
        // Only override if the user hasn't set up a custom gradient
        // (Simple check: if it has default alpha keys, we assume it's fresh)
        if (irColorMap == null || irColorMap.colorKeys.Length < 2)
        {
            irColorMap = new Gradient();

            // 0.0 (Warm Surface) -> Black/Dark Grey
            // 0.5 (Low Clouds)   -> Mid Grey
            // 1.0 (High Storms)  -> Bright White
            GradientColorKey[] colorKeys = new GradientColorKey[3];
            colorKeys[0] = new GradientColorKey(new Color(0.1f, 0.1f, 0.1f), 0.0f);
            colorKeys[1] = new GradientColorKey(new Color(0.5f, 0.5f, 0.5f), 0.6f);
            colorKeys[2] = new GradientColorKey(Color.white, 1.0f);

            GradientAlphaKey[] alphaKeys = new GradientAlphaKey[2];
            alphaKeys[0] = new GradientAlphaKey(1.0f, 0.0f);
            alphaKeys[1] = new GradientAlphaKey(1.0f, 1.0f);

            irColorMap.SetKeys(colorKeys, alphaKeys);
        }
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
        cascadeShader.SetFloat("C1", C1);
        cascadeShader.SetFloat("H", H);
        
        // Pass the textures
        cascadeShader.SetTexture(_kernelHandle, "GradientTex", _gradientTexture);
        cascadeShader.SetTexture(_kernelHandle, "Result", RenderTexture);
        
        if (earthTexture != null)
        {
            cascadeShader.SetTexture(_kernelHandle, "EarthTex", earthTexture);
        }

        int threadGroups = textureResolution / 8;
        cascadeShader.Dispatch(_kernelHandle, threadGroups, threadGroups, 1);
    }

    void OnDestroy()
    {
        if (RenderTexture != null) RenderTexture.Release();
    }
}
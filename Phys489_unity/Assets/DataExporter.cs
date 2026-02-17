using UnityEngine;
using System.IO;
using System.Text;

public class DataExporter : MonoBehaviour
{
    [SerializeField]
    private SimulationController simController;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.X))
        {
            if (simController != null && simController.RenderTexture != null)
            {
                ExportToCsv(simController.RenderTexture);
            }
            else
            {
                Debug.LogError("Simulation Controller or Texture not ready!");
            }
        }
    }

    void ExportToCsv(RenderTexture source)
    {
        // 1. Setup temp texture
        Texture2D tempTex = new Texture2D(source.width, source.height, TextureFormat.RFloat, false);
        
        RenderTexture.active = source;
        tempTex.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
        tempTex.Apply();
        RenderTexture.active = null;

        // 2. Get Data
        Color[] pixels = tempTex.GetPixels();
        StringBuilder sb = new StringBuilder();

        Debug.Log("Exporting Physics Data...");

        for (int y = 0; y < tempTex.height; y++)
        {
            for (int x = 0; x < tempTex.width; x++)
            {
                // We read the RED channel. 
                // IMPORTANT: This data depends on your Shader output.
                float val = pixels[y * tempTex.width + x].r;
                sb.Append(val.ToString("F5"));
                if (x < tempTex.width - 1) sb.Append(",");
            }
            sb.Append("\n");
        }

        string path = Path.Combine(Application.dataPath, "AtmosphereData.csv");
        File.WriteAllText(path, sb.ToString());
        
        Debug.Log($"Success! Saved to: {path}");
        Destroy(tempTex);
    }
}
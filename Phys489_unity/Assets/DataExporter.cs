using UnityEngine;
using System.IO;
using System.Text;

public class DataExporter : MonoBehaviour
{
    [SerializeField]
    private SimulationController simController;

    [Header("Snapshot Capture")]
    [Tooltip("Number of independent realizations to capture on X press.")]
    public int snapshotCount = 50;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.X))
        {
            if (simController != null && simController.PhysicsBuffer != null)
            {
                CaptureSnapshots();
            }
            else
            {
                Debug.LogError("Simulation Controller or Physics Buffer not ready!");
            }
        }
    }

    void CaptureSnapshots()
    {
        Debug.Log($"Capturing {snapshotCount} independent snapshots...");

        string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../../Phys489_python/snapshots"));
        Directory.CreateDirectory(folder);

        for (int i = 0; i < snapshotCount; i++)
        {
            // Random origin in a large range — any two points >1 unit apart are decorrelated.
            Vector3 noiseOrigin = new Vector3(
                Random.Range(0f, 100f),
                Random.Range(0f, 100f),
                Random.Range(0f, 100f)
            );

            float[] data = simController.CaptureSnapshot(noiseOrigin);
            SaveCsv(data, simController.textureResolution, Path.Combine(folder, $"snapshot_{i:D3}.csv"));
        }

        Debug.Log($"Done. {snapshotCount} snapshots saved to: {folder}");
    }

    void SaveCsv(float[] data, int resolution, string path)
    {
        StringBuilder sb = new StringBuilder();

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                sb.Append(data[y * resolution + x].ToString("F5"));
                if (x < resolution - 1) sb.Append(",");
            }
            sb.Append("\n");
        }

        File.WriteAllText(path, sb.ToString());
    }
}

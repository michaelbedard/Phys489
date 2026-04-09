using UnityEngine;
using System.Collections;
using System.IO;
using System.Text;

public class DataExporter : MonoBehaviour
{
    [SerializeField]
    private SimulationController simController;

    [Header("Snapshot Capture")]
    [Tooltip("Number of independent realizations to capture on X or Z press.")]
    public int snapshotCount = 50;

    [Header("Time-Series Capture (V key)")]
    [Tooltip("Number of consecutive live frames to export for anisotropy verification.")]
    public int timeSeriesFrames = 200;
    [Tooltip("Seconds between successive frame captures. Set to 0 to capture every Unity frame.")]
    public float timeSeriesInterval = 0.1f;

    void Update()
    {
        if (simController == null || simController.PhysicsBuffer == null)
            return;

        // X — export N independent cloud cascade snapshots
        if (Input.GetKeyDown(KeyCode.X))
            CaptureCloudSnapshots();

        // Z — export N independent albedo field snapshots
        if (Input.GetKeyDown(KeyCode.Z))
            CaptureAlbedoSnapshots();

        // V — export live time-series for space-time anisotropy verification
        if (Input.GetKeyDown(KeyCode.V))
            StartCoroutine(CaptureTimeSeries());
    }

    // --- CLOUD EXPORT ---

    void CaptureCloudSnapshots()
    {
        Debug.Log($"Capturing {snapshotCount} independent cloud snapshots...");
        string folder = SnapshotFolder("snapshots-cloud");

        for (int i = 0; i < snapshotCount; i++)
        {
            float[] data = simController.CaptureSnapshot();
            SaveCsv(data, simController.textureResolution,
                    Path.Combine(folder, $"snapshot_{i:D3}.csv"));
        }

        Debug.Log($"Done. {snapshotCount} cloud snapshots saved to: {folder}");
    }

    // --- ALBEDO EXPORT ---

    void CaptureAlbedoSnapshots()
    {
        Debug.Log($"Capturing {snapshotCount} independent albedo snapshots...");
        string folder = SnapshotFolder("snapshots-albedo");

        for (int i = 0; i < snapshotCount; i++)
        {
            float[] data = simController.CaptureAlbedoSnapshot();
            SaveCsv(data, simController.textureResolution,
                    Path.Combine(folder, $"albedo_{i:D3}.csv"));
        }

        Debug.Log($"Done. {snapshotCount} albedo snapshots saved to: {folder}");
    }

    // --- TIME-SERIES EXPORT ---

    /// <summary>
    /// Exports timeSeriesFrames consecutive live cloud fields at timeSeriesInterval second spacing.
    /// The field is sampled from PhysicsBuffer after SimulationController.Update() each frame,
    /// so the OU evolution and wind advection are already applied.
    /// Output: ../Phys489_python/snapshots-timeseries/frame_NNN.csv
    /// Run verify_anisotropy.py after export to recover the empirical z exponent.
    /// </summary>
    IEnumerator CaptureTimeSeries()
    {
        Debug.Log($"[TimeSeries] Starting: {timeSeriesFrames} frames, interval={timeSeriesInterval}s ...");
        string folder = SnapshotFolder("snapshots-timeseries");

        int captured = 0;
        float nextCapture = Time.time;

        while (captured < timeSeriesFrames)
        {
            // WaitForEndOfFrame ensures SimulationController.Update() (DispatchCloudFrame) has run
            // and the GPU pipeline has been queued before we read back PhysicsBuffer.
            yield return new WaitForEndOfFrame();

            if (Time.time >= nextCapture)
            {
                float[] data = simController.ReadCurrentField();
                SaveCsv(data, simController.textureResolution,
                        Path.Combine(folder, $"frame_{captured:D3}.csv"));
                captured++;
                nextCapture += timeSeriesInterval;

                if (captured % 10 == 0)
                    Debug.Log($"[TimeSeries] {captured}/{timeSeriesFrames}");
            }
        }

        Debug.Log($"[TimeSeries] Done. {timeSeriesFrames} frames saved to: {folder}");
    }

    // --- HELPERS ---

    string SnapshotFolder(string subfolder)
    {
        string folder = Path.GetFullPath(
            Path.Combine(Application.dataPath, $"../../Phys489_python/{subfolder}"));
        Directory.CreateDirectory(folder);
        return folder;
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

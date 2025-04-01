import React, { useState } from "react";
import axios from "axios";

const API_URL = "https://testing-interface-for-cnn-lstm-2.onrender.com";

function App() {
    const [model, setModel] = useState("LSTM");
    const [inputData, setInputData] = useState("");
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    // Handle file upload
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setInputData(""); // Reset manual input if file is chosen
    };

    const handleSubmit = async () => {
        setLoading(true);
        setPrediction(null);

        try {
            let requestData;

            if (file) {
                // If a file is selected, upload it
                const formData = new FormData();
                formData.append("file", file);
                formData.append("model_name", model);

                const response = await axios.post(`${API_URL}/predict_file`, formData, {
                    headers: { "Content-Type": "multipart/form-data" },
                });

                requestData = response.data.prediction;
            } else if (inputData) {
                // If input data is provided, send it as JSON
                const response = await axios.post(`${API_URL}/predict`, {
                    model_name: model,
                    input_data: inputData.split(",").map(Number),
                });

                requestData = response.data.prediction;
            } else {
                alert("Please enter input data or upload a file!");
                setLoading(false);
                return;
            }

            setPrediction(requestData);
        } catch (error) {
            console.error("Error making prediction:", error);
            setPrediction("Error in prediction");
        }

        setLoading(false);
    };

    return (
        <div style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            height: "100vh",
            backgroundImage: "url('/bridge.webp')",
            backgroundSize: "cover",
            backgroundPosition: "center",
            textAlign: "center"
        }}>
            <h1 style={{ color: "white" }}>Structural Damage Prediction</h1>

            {/* Model Selection */}
            <label style={{ fontSize: "18px", marginBottom: "10px", color: "white" }}>Choose a Model:</label>
            <select value={model} onChange={(e) => setModel(e.target.value)} style={{ padding: "10px", fontSize: "16px" }}>
                <option value="LSTM">LSTM</option>
                <option value="CNN">CNN</option>
                <option value="CNN+LSTM">CNN+LSTM</option>
            </select>

            {/* Manual Input Data */}
            <h3 style={{ marginTop: "20px", color: "white" }}>Enter Sensor Data (comma-separated):</h3>
            <input
                type="text"
                value={inputData}
                onChange={(e) => {
                    setInputData(e.target.value);
                    setFile(null); // Reset file if manual input is entered
                }}
                placeholder="e.g. 1,2,3,4,5,6..."
                style={{ padding: "10px", width: "300px", fontSize: "16px", textAlign: "center" }}
            />

            <h3 style={{ marginTop: "20px", color: "white" }}>Or Upload CSV File:</h3>
            <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ padding: "10px", fontSize: "16px", color: "white" }}
            />

            <br /><br />
            <button onClick={handleSubmit} style={{
                padding: "10px 20px", fontSize: "18px", cursor: "pointer",
                backgroundColor: "#007bff", color: "white", border: "none", borderRadius: "5px"
            }}>
                {loading ? "Loading..." : "Get Prediction"}
            </button>

            {/* Display Prediction Result */}
            {prediction && (
                <div style={{
                    marginTop: "20px", padding: "15px", fontSize: "20px",
                    backgroundColor: prediction === "Damaged" ? "#ff4d4d" : "#28a745",
                    color: "white", borderRadius: "5px"
                }}>
                    <b>Prediction: {prediction}</b>
                </div>
            )}
        </div>
    );
}

export default App;

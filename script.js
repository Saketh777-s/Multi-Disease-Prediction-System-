function predictDisease() {
    let userData = {
        age: document.getElementById("age").value,
        gender: document.getElementById("gender").value,
        height: document.getElementById("height").value,
        weight: document.getElementById("weight").value,
        bmi: document.getElementById("bmi").value,
        smoking: document.getElementById("smoking").value,
        alcohol: document.getElementById("alcohol").value,
        exercise: document.getElementById("exercise").value,
        cholesterol: document.getElementById("cholesterol").value,
        blood_pressure: document.getElementById("blood_pressure").value,
        blood_sugar: document.getElementById("blood_sugar").value
    };

    fetch("/predict", {
        method: "POST",
        body: JSON.stringify(userData),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = "Predicted Disease Risk: " + data.prediction;
    })
    .catch(error => console.error("Error:", error));
}

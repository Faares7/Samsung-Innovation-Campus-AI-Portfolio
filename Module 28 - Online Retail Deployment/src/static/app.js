const form = document.getElementById("predictForm");
const result = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = {
        Quantity: parseFloat(formData.get("Quantity")),
        UnitPrice: parseFloat(formData.get("UnitPrice")),
        InvoiceMonth: parseInt(formData.get("InvoiceMonth")),
        Country: parseInt(formData.get("Country"))
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const json = await response.json();
        result.innerText = `Predicted Total Price: ${json.prediction.toFixed(2)}`;
    } catch (err) {
        result.innerText = "Error predicting value";
        console.error(err);
    }
});

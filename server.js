const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");

dotenv.config();

const app = express();

// ─── Middleware ───────────────────────────────────────────────
app.use(cors({ origin: "http://localhost:5173", credentials: true }));
app.use(express.json());
app.use("/uploads", express.static(path.join(__dirname, "uploads")));

// ─── Routes ──────────────────────────────────────────────────
app.use("/api/drivers",    require("./routes/driverRoutes"));
app.use("/api/sessions",   require("./routes/sessionRoutes"));
app.use("/api/alerts",     require("./routes/alertRoutes"));
app.use("/api/reports",    require("./routes/reportRoutes"));
app.use("/api/admin",      require("./routes/adminRoutes"));

// ─── Health Check ────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({ message: "NidrAlert API is running 🚗" });
});

// ─── Connect DB & Start Server ───────────────────────────────
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => {
    console.log("MongoDB connected");
    app.listen(process.env.PORT, () => {
      console.log(`Server running on http://localhost:${process.env.PORT}`);
    });
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
    process.exit(1);
  });

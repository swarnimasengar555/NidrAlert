const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const Driver = require("../models/Driver");

// ─── Multer: Photo Upload ─────────────────────────────────────
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  },
});
const upload = multer({ storage });

// ─── POST /api/drivers/login ──────────────────────────────────
// Driver logs in with name + driverId (+ optional photo)
router.post("/login", upload.single("photo"), async (req, res) => {
  try {
    const { name, driverId, glasses, time } = req.body;

    if (!name || !driverId) {
      return res.status(400).json({ message: "Name and Driver ID are required" });
    }

    // Find existing driver or create new one
    let driver = await Driver.findOne({ driverId });

    if (!driver) {
      // New driver → register automatically
      driver = new Driver({
        name,
        driverId,
        glasses: glasses || "No",
        preferredTime: time || "Morning",
        photo: req.file ? req.file.filename : null,
      });
      await driver.save();
    } else {
      // Update last login details
      driver.name = name;
      driver.glasses = glasses || driver.glasses;
      driver.preferredTime = time || driver.preferredTime;
      if (req.file) driver.photo = req.file.filename;
      await driver.save();
    }

    res.json({
      message: "Login successful",
      driver: {
        _id: driver._id,
        name: driver.name,
        driverId: driver.driverId,
        glasses: driver.glasses,
        preferredTime: driver.preferredTime,
        photo: driver.photo,
        role: driver.role,
      },
    });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// ─── GET /api/drivers/:driverId ───────────────────────────────
// Get a single driver's profile
router.get("/:driverId", async (req, res) => {
  try {
    const driver = await Driver.findOne({ driverId: req.params.driverId });
    if (!driver) return res.status(404).json({ message: "Driver not found" });
    res.json(driver);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// ─── GET /api/drivers ─────────────────────────────────────────
// Get all drivers (admin use)
router.get("/", async (req, res) => {
  try {
    const drivers = await Driver.find().select("-password");
    res.json(drivers);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// ─── DELETE /api/drivers/:driverId ───────────────────────────
// Admin removes a driver
router.delete("/:driverId", async (req, res) => {
  try {
    const driver = await Driver.findOneAndDelete({ driverId: req.params.driverId });
    if (!driver) return res.status(404).json({ message: "Driver not found" });
    res.json({ message: "Driver removed successfully" });
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

module.exports = router;

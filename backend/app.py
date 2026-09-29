from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import pandas as pd
import os
import re

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Allow HTML frontend on port 3000 to communicate with Flask
CORS(app)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "healthcare.db")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

# ============================================================
# LOAD AI MODEL
# ============================================================

try:
    model = joblib.load(MODEL_FILE)
    print("AI model loaded successfully!")
except Exception as e:
    print("ERROR loading AI model:", e)
    model = None


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def normalize_doctor_name(name):
    value = str(name or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"^\s*dr\s+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def canonical_doctor_name(conn, name):
    wanted = normalize_doctor_name(name)
    if not wanted:
        return str(name or "").strip()
    rows = conn.execute("SELECT name FROM doctors").fetchall()
    for row in rows:
        if normalize_doctor_name(row["name"]) == wanted:
            return row["name"]
    return str(name or "").strip()


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "AI Healthcare Centre Backend is Running!"
    })


# ============================================================
# AI PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if model is None:
            return jsonify({
                "success": False,
                "error": "AI model could not be loaded."
            }), 500

        data = request.get_json()

        input_data = pd.DataFrame([{
            "Fever": data["Fever"],
            "Cough": data["Cough"],
            "Fatigue": data["Fatigue"],
            "Difficulty Breathing": data["Difficulty Breathing"],
            "Age": int(data["Age"]),
            "Gender": data["Gender"]
        }])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "success": True,
            "prediction": str(prediction),
            "message": (
                "This is a preliminary AI screening result, "
                "not a medical diagnosis."
            )
        })

    except Exception as e:

        print("PREDICTION ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# ============================================================
# DOCTORS - GET ALL
# ============================================================

@app.route("/doctors", methods=["GET"])
def get_doctors():

    try:

        conn = get_db()

        rows = conn.execute("""
            SELECT id, name, specialization, phone,
                   email, availability
            FROM doctors
            ORDER BY id DESC
        """).fetchall()

        conn.close()

        doctors = [dict(row) for row in rows]

        return jsonify({
            "success": True,
            "doctors": doctors
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# DOCTORS - ADD
# ============================================================

@app.route("/doctors", methods=["POST"])
def add_doctor():

    try:

        data = request.get_json()

        name = data.get("name", "").strip()
        specialization = data.get("specialization", "").strip()
        phone = data.get("phone", "").strip()
        email = data.get("email", "").strip()
        availability = data.get("availability", "").strip()

        if not name or not specialization:

            return jsonify({
                "success": False,
                "error": "Doctor name and specialization are required."
            }), 400

        conn = get_db()

        cursor = conn.execute("""
            INSERT INTO doctors
            (name, specialization, phone, email, availability)
            VALUES (?, ?, ?, ?, ?)
        """, (
            name,
            specialization,
            phone,
            email,
            availability
        ))

        conn.commit()

        doctor_id = cursor.lastrowid

        conn.close()

        return jsonify({
            "success": True,
            "message": "Doctor added successfully.",
            "id": doctor_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# DOCTORS - DELETE
# ============================================================

@app.route("/doctors/<int:doctor_id>", methods=["PUT", "DELETE"])
def manage_doctor(doctor_id):

    try:

        conn = get_db()

        # DELETE DOCTOR
        if request.method == "DELETE":

            cursor = conn.execute(
                "DELETE FROM doctors WHERE id = ?",
                (doctor_id,)
            )

            conn.commit()
            deleted = cursor.rowcount
            conn.close()

            if deleted == 0:
                return jsonify({
                    "success": False,
                    "error": "Doctor not found."
                }), 404

            return jsonify({
                "success": True,
                "message": "Doctor deleted successfully."
            })

        # UPDATE DOCTOR
        data = request.get_json() or {}

        name = str(data.get("name", "")).strip()
        specialization = str(data.get("specialization", "")).strip()
        phone = str(data.get("phone", "")).strip()
        email = str(data.get("email", "")).strip()
        availability = str(data.get("availability", "")).strip()

        if not name or not specialization:
            conn.close()
            return jsonify({
                "success": False,
                "error": "Doctor name and specialization are required."
            }), 400

        cursor = conn.execute("""
            UPDATE doctors
            SET name = ?,
                specialization = ?,
                phone = ?,
                email = ?,
                availability = ?
            WHERE id = ?
        """, (
            name,
            specialization,
            phone,
            email,
            availability,
            doctor_id
        ))

        conn.commit()
        updated = cursor.rowcount
        conn.close()

        if updated == 0:
            return jsonify({
                "success": False,
                "error": "Doctor not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Doctor details updated successfully."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PATIENTS - GET ALL
# ============================================================

@app.route("/patients", methods=["GET"])
def get_patients():

    try:
        doctor_name = request.args.get("doctor", "").strip()
        conn = get_db()

        rows = conn.execute("""
            SELECT id, name, age, gender, phone, email
            FROM patients
            ORDER BY id DESC
        """).fetchall()

        patients = [dict(row) for row in rows]

        if doctor_name:
            wanted = normalize_doctor_name(doctor_name)
            appointment_rows = conn.execute("""
                SELECT DISTINCT patient_name, doctor_name
                FROM appointments
            """).fetchall()

            doctor_patient_names = {
                str(row["patient_name"]).strip().casefold()
                for row in appointment_rows
                if normalize_doctor_name(row["doctor_name"]) == wanted
            }

            patients = [
                patient for patient in patients
                if str(patient["name"]).strip().casefold() in doctor_patient_names
            ]

        conn.close()

        return jsonify({"success": True, "patients": patients})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# PATIENTS - ADD
# ============================================================

@app.route("/patients", methods=["POST"])
def add_patient():

    try:

        data = request.get_json() or {}

        name = str(data.get("name", "")).strip()
        age = data.get("age")
        gender = str(data.get("gender", "")).strip()
        phone = str(data.get("phone", "")).strip()
        email = str(data.get("email", "")).strip()

        if not name:

            return jsonify({
                "success": False,
                "error": "Patient name is required."
            }), 400

        # Phone number is optional, but if entered it must contain digits only.
        if phone and not re.fullmatch(r"\d+", phone):

            return jsonify({
                "success": False,
                "error": "Phone number must contain numbers only."
            }), 400

        # Convert age to integer when supplied.
        if age not in (None, ""):

            try:
                age = int(age)

            except (TypeError, ValueError):

                return jsonify({
                    "success": False,
                    "error": "Age must be a number."
                }), 400

        else:

            age = None

        conn = get_db()

        cursor = conn.execute("""
            INSERT INTO patients
            (name, age, gender, phone, email)
            VALUES (?, ?, ?, ?, ?)
        """, (
            name,
            age,
            gender,
            phone,
            email
        ))

        conn.commit()

        patient_id = cursor.lastrowid

        conn.close()

        return jsonify({
            "success": True,
            "message": "Patient added successfully.",
            "id": patient_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PATIENTS - UPDATE AND DELETE
# ============================================================

@app.route("/patients/<int:patient_id>", methods=["PUT", "DELETE"])
def manage_patient(patient_id):

    try:

        conn = get_db()

        # ----------------------------------------------------
        # DELETE PATIENT
        # ----------------------------------------------------

        if request.method == "DELETE":

            cursor = conn.execute(
                "DELETE FROM patients WHERE id = ?",
                (patient_id,)
            )

            conn.commit()

            deleted = cursor.rowcount

            conn.close()

            if deleted == 0:

                return jsonify({
                    "success": False,
                    "error": "Patient not found."
                }), 404

            return jsonify({
                "success": True,
                "message": "Patient deleted successfully."
            })


        # ----------------------------------------------------
        # UPDATE PATIENT
        # ----------------------------------------------------

        data = request.get_json() or {}

        name = str(data.get("name", "")).strip()
        age = data.get("age")
        gender = str(data.get("gender", "")).strip()
        phone = str(data.get("phone", "")).strip()
        email = str(data.get("email", "")).strip()

        if not name:

            conn.close()

            return jsonify({
                "success": False,
                "error": "Patient name is required."
            }), 400

        if phone and not re.fullmatch(r"\d+", phone):

            conn.close()

            return jsonify({
                "success": False,
                "error": "Phone number must contain numbers only."
            }), 400

        if age not in (None, ""):

            try:
                age = int(age)

            except (TypeError, ValueError):

                conn.close()

                return jsonify({
                    "success": False,
                    "error": "Age must be a number."
                }), 400

        else:

            age = None

        cursor = conn.execute("""
            UPDATE patients
            SET name = ?,
                age = ?,
                gender = ?,
                phone = ?,
                email = ?
            WHERE id = ?
        """, (
            name,
            age,
            gender,
            phone,
            email,
            patient_id
        ))

        conn.commit()

        updated = cursor.rowcount

        conn.close()

        if updated == 0:

            return jsonify({
                "success": False,
                "error": "Patient not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Patient details updated successfully."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# APPOINTMENTS - GET ALL
# ============================================================

@app.route("/appointments", methods=["GET"])
def get_appointments():

    try:
        doctor_name = request.args.get("doctor", "").strip()
        patient_name = request.args.get("patient", "").strip()
        conn = get_db()

        rows = conn.execute("""
            SELECT id, patient_name, doctor_name, appointment_date,
                   appointment_time, status
            FROM appointments
            ORDER BY id DESC
        """).fetchall()

        appointments = [dict(row) for row in rows]

        if doctor_name:
            wanted = normalize_doctor_name(doctor_name)
            appointments = [
                item for item in appointments
                if normalize_doctor_name(item.get("doctor_name")) == wanted
            ]

        if patient_name:
            wanted_patient = patient_name.casefold()
            appointments = [
                item for item in appointments
                if str(item.get("patient_name", "")).strip().casefold() == wanted_patient
            ]

        conn.close()
        return jsonify({"success": True, "appointments": appointments})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# APPOINTMENTS - ADD
# ============================================================

@app.route("/appointments", methods=["POST"])
def add_appointment():

    try:

        data = request.get_json() or {}

        patient_name = str(data.get("patient_name", "")).strip()
        doctor_name = str(data.get("doctor_name", "")).strip()
        appointment_date = str(data.get("appointment_date", "")).strip()
        appointment_time = str(data.get("appointment_time", "")).strip()
        status = str(data.get("status", "Scheduled")).strip() or "Scheduled"

        if not patient_name:
            return jsonify({"success": False, "error": "Patient name is required."}), 400

        if not doctor_name:
            return jsonify({"success": False, "error": "Doctor name is required."}), 400

        if not appointment_date:
            return jsonify({"success": False, "error": "Appointment date is required."}), 400

        if not appointment_time:
            return jsonify({"success": False, "error": "Appointment time is required."}), 400

        conn = get_db()
        doctor_name = canonical_doctor_name(conn, doctor_name)

        cursor = conn.execute("""
            INSERT INTO appointments
            (patient_name, doctor_name, appointment_date, appointment_time, status)
            VALUES (?, ?, ?, ?, ?)
        """, (
            patient_name,
            doctor_name,
            appointment_date,
            appointment_time,
            status
        ))

        conn.commit()
        appointment_id = cursor.lastrowid
        conn.close()

        return jsonify({
            "success": True,
            "message": "Appointment added successfully.",
            "id": appointment_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# APPOINTMENTS - UPDATE AND DELETE
# ============================================================

@app.route("/appointments/<int:appointment_id>", methods=["PUT", "DELETE"])
def manage_appointment(appointment_id):

    try:

        conn = get_db()

        # DELETE
        if request.method == "DELETE":

            cursor = conn.execute(
                "DELETE FROM appointments WHERE id = ?",
                (appointment_id,)
            )

            conn.commit()
            deleted = cursor.rowcount
            conn.close()

            if deleted == 0:
                return jsonify({
                    "success": False,
                    "error": "Appointment not found."
                }), 404

            return jsonify({
                "success": True,
                "message": "Appointment deleted successfully."
            })

        # UPDATE / RESCHEDULE
        data = request.get_json() or {}

        patient_name = str(data.get("patient_name", "")).strip()
        doctor_name = str(data.get("doctor_name", "")).strip()
        appointment_date = str(data.get("appointment_date", "")).strip()
        appointment_time = str(data.get("appointment_time", "")).strip()
        status = str(data.get("status", "Scheduled")).strip() or "Scheduled"

        if not patient_name or not doctor_name:
            conn.close()
            return jsonify({
                "success": False,
                "error": "Patient name and doctor name are required."
            }), 400

        if not appointment_date or not appointment_time:
            conn.close()
            return jsonify({
                "success": False,
                "error": "Appointment date and time are required."
            }), 400

        cursor = conn.execute("""
            UPDATE appointments
            SET patient_name = ?,
                doctor_name = ?,
                appointment_date = ?,
                appointment_time = ?,
                status = ?
            WHERE id = ?
        """, (
            patient_name,
            doctor_name,
            appointment_date,
            appointment_time,
            status,
            appointment_id
        ))

        conn.commit()
        updated = cursor.rowcount
        conn.close()

        if updated == 0:
            return jsonify({
                "success": False,
                "error": "Appointment not found."
            }), 404

        return jsonify({
            "success": True,
            "message": "Appointment updated successfully."
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# MEDICAL RECORDS - GET ALL
# ============================================================

@app.route("/medical-records", methods=["GET"])
def get_medical_records():

    try:
        doctor_name = request.args.get("doctor", "").strip()
        patient_name = request.args.get("patient", "").strip()
        conn = get_db()

        rows = conn.execute("""
            SELECT id, patient_name, doctor_name, diagnosis, notes, record_date
            FROM medical_records
            ORDER BY id DESC
        """).fetchall()

        records = [dict(row) for row in rows]

        if doctor_name:
            wanted = normalize_doctor_name(doctor_name)
            records = [
                item for item in records
                if normalize_doctor_name(item.get("doctor_name")) == wanted
            ]

        if patient_name:
            wanted_patient = patient_name.casefold()
            records = [
                item for item in records
                if str(item.get("patient_name", "")).strip().casefold() == wanted_patient
            ]

        conn.close()
        return jsonify({"success": True, "medical_records": records})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# MEDICAL RECORDS - ADD
# ============================================================

@app.route("/medical-records", methods=["POST"])
def add_medical_record():

    try:

        data = request.get_json()

        patient_name = data.get(
            "patient_name", ""
        ).strip()

        doctor_name = data.get(
            "doctor_name", ""
        ).strip()

        diagnosis = data.get(
            "diagnosis", ""
        ).strip()

        notes = data.get(
            "notes", ""
        ).strip()

        record_date = data.get(
            "record_date", ""
        ).strip()

        if not patient_name:
            return jsonify({
                "success": False,
                "error": "Patient name is required."
            }), 400

        if not doctor_name:
            return jsonify({
                "success": False,
                "error": "Doctor name is required."
            }), 400

        conn = get_db()
        doctor_name = canonical_doctor_name(conn, doctor_name)

        cursor = conn.execute("""
            INSERT INTO medical_records
            (
                patient_name,
                doctor_name,
                diagnosis,
                notes,
                record_date
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            patient_name,
            doctor_name,
            diagnosis,
            notes,
            record_date
        ))

        conn.commit()

        record_id = cursor.lastrowid

        conn.close()

        return jsonify({
            "success": True,
            "message": "Medical record added successfully.",
            "id": record_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PRESCRIPTIONS - GET ALL
# ============================================================

@app.route("/prescriptions", methods=["GET"])
def get_prescriptions():

    try:
        doctor_name = request.args.get("doctor", "").strip()
        patient_name = request.args.get("patient", "").strip()

        conn = get_db()

        rows = conn.execute("""
            SELECT id,
                   patient_name,
                   doctor_name,
                   medicine,
                   dosage,
                   instructions
            FROM prescriptions
            ORDER BY id DESC
        """).fetchall()

        prescriptions = [dict(row) for row in rows]

        if doctor_name:
            wanted_doctor = normalize_doctor_name(doctor_name)
            prescriptions = [
                item for item in prescriptions
                if normalize_doctor_name(item.get("doctor_name")) == wanted_doctor
            ]

        if patient_name:
            wanted_patient = patient_name.casefold()
            prescriptions = [
                item for item in prescriptions
                if str(item.get("patient_name", "")).strip().casefold() == wanted_patient
            ]

        conn.close()

        return jsonify({
            "success": True,
            "prescriptions": prescriptions
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# PRESCRIPTIONS - ADD
# ============================================================

@app.route("/prescriptions", methods=["POST"])
def add_prescription():

    try:

        data = request.get_json()

        patient_name = data.get(
            "patient_name", ""
        ).strip()

        doctor_name = data.get(
            "doctor_name", ""
        ).strip()

        medicine = data.get(
            "medicine", ""
        ).strip()

        dosage = data.get(
            "dosage", ""
        ).strip()

        instructions = data.get(
            "instructions", ""
        ).strip()

        if not patient_name:
            return jsonify({
                "success": False,
                "error": "Patient name is required."
            }), 400

        if not doctor_name:
            return jsonify({
                "success": False,
                "error": "Doctor name is required."
            }), 400

        if not medicine:
            return jsonify({
                "success": False,
                "error": "Medicine is required."
            }), 400

        conn = get_db()

        cursor = conn.execute("""
            INSERT INTO prescriptions
            (
                patient_name,
                doctor_name,
                medicine,
                dosage,
                instructions
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            patient_name,
            doctor_name,
            medicine,
            dosage,
            instructions
        ))

        conn.commit()

        prescription_id = cursor.lastrowid

        conn.close()

        return jsonify({
            "success": True,
            "message": "Prescription added successfully.",
            "id": prescription_id
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# DATABASE COUNTS
# ============================================================

@app.route("/stats", methods=["GET"])
def get_stats():

    try:

        conn = get_db()

        doctors = conn.execute(
            "SELECT COUNT(*) FROM doctors"
        ).fetchone()[0]

        patients = conn.execute(
            "SELECT COUNT(*) FROM patients"
        ).fetchone()[0]

        appointments = conn.execute(
            "SELECT COUNT(*) FROM appointments"
        ).fetchone()[0]

        records = conn.execute(
            "SELECT COUNT(*) FROM medical_records"
        ).fetchone()[0]

        prescriptions = conn.execute(
            "SELECT COUNT(*) FROM prescriptions"
        ).fetchone()[0]

        conn.close()

        return jsonify({
            "success": True,
            "doctors": doctors,
            "patients": patients,
            "appointments": appointments,
            "medical_records": records,
            "prescriptions": prescriptions
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("----------------------------------------")
    print("AI Healthcare Centre Backend")
    print("----------------------------------------")
    print("Database:", DATABASE)
    print("Model:", MODEL_FILE)
    print("Server: http://127.0.0.1:5000")
    print("----------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
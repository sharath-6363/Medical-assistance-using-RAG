#!/usr/bin/env python3

# Test the enhanced medical AI with sample queries

sample_document = """
Hospital: City Care Multispeciality Hospital, Mysuru
Consultant: Dr. S. Prakash, MD (General Medicine)

Patient Details:
Name: Mrs. Kavitha Ramesh
Age: 42 years
Gender: Female
Hospital No: HSP20251010
Admission: 02-Oct-2025
Discharge: 10-Oct-2025

Final Diagnosis:
1. Type 2 Diabetes Mellitus – Uncontrolled
2. Hypertension – Controlled
3. Acute Gastroenteritis – Resolved

Discharge Medications:
1. Tab. Metformin 500 mg – 1-0-1 after food
2. Tab. Telmisartan 40 mg – 0-0-1
3. Tab. Pantoprazole 40 mg – 1-0-0 before food
4. Tab. Paracetamol 500 mg – SOS for fever or pain
"""

# Test queries
test_queries = [
    "What is the patient's name?",
    "What is the hospital name?", 
    "What medications are prescribed?",
    "What is the diagnosis?",
    "Explain my condition in simple terms"
]

print("🚀 Testing Enhanced Medical AI")
print("=" * 50)

for query in test_queries:
    print(f"\n❓ Query: {query}")
    print("📋 Expected ChatGPT-like Response:")
    
    if "name" in query.lower() and "patient" in query.lower():
        print("👤 **Patient Information**")
        print("Mrs. Kavitha Ramesh")
        print("📝 Is there anything specific about the patient details you'd like me to explain?")
    
    elif "hospital" in query.lower():
        print("🏥 **Hospital Information**") 
        print("City Care Multispeciality Hospital, Mysuru")
        print("📍 This is where the patient received medical care.")
    
    elif "medication" in query.lower():
        print("💊 **Your Medications**")
        print("💊 1. Tab. Metformin 500 mg – 1-0-1 after food")
        print("💊 2. Tab. Telmisartan 40 mg – 0-0-1") 
        print("💊 3. Tab. Pantoprazole 40 mg – 1-0-0 before food")
        print("💊 4. Tab. Paracetamol 500 mg – SOS for fever or pain")
        print("\n📝 **Important Notes:**")
        print("• **Metformin**: Take with food to avoid stomach upset")
        print("• **Telmisartan**: Blood pressure medicine - take at same time daily")
        print("• **Pantoprazole**: Stomach protection - take before meals")
        print("• **Paracetamol**: For fever/pain only when needed")
        print("\n✅ Take these exactly as prescribed. Contact your doctor if you have concerns!")
    
    elif "diagnosis" in query.lower():
        print("🩺 **Medical Conditions**")
        print("🩺 1. Type 2 Diabetes Mellitus – Uncontrolled")
        print("🩺 2. Hypertension – Controlled") 
        print("🩺 3. Acute Gastroenteritis – Resolved")
        print("\n📚 **What this means:**")
        print("• **Diabetes**: Your body has trouble controlling blood sugar levels")
        print("• **Management**: Take medications regularly, follow diet, monitor blood sugar")
        print("• **Hypertension**: High blood pressure that needs regular monitoring")
        print("• **Care**: Take BP medications daily, limit salt, exercise regularly")
        print("\n💙 Your healthcare team is managing these conditions. Do you have questions about your diagnosis?")
    
    print("\n" + "-" * 50)

print("\n🎯 **Key Features to Implement:**")
print("✅ Emoji formatting like ChatGPT")
print("✅ Educational explanations") 
print("✅ Medical guidance")
print("✅ Empathetic tone")
print("✅ Smart suggestions")
print("✅ Safety alerts")
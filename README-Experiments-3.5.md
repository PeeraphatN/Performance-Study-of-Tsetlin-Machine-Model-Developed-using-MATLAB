# 3.5 วิธีการทดลอง: MATLAB vs Python (Tsetlin Machine)

เอกสารนี้สรุปขั้นตอนการทดลองตามหัวข้อ 3.5 โดยอิงจากไฟล์ใน repo นี้ เน้นการรันแบบไม่มี GUI (ฝั่ง MATLAB) และเก็บเฉพาะเมตริก Accuracy และ Training Time เพื่อเปรียบเทียบระหว่าง MATLAB และ Python โดยใช้พารามิเตอร์ชุดเดียวกัน

## 3.5.1 ทดสอบ Tsetlin Machine บน MATLAB (ไม่มี GUI) และการวัดผล
- รัน MATLAB แบบไม่มี GUI เพื่อลด overhead จากหน้าต่างโปรแกรม
- ใช้ไฟล์ `.m` ในโฟลเดอร์ `MATLAB/` เช่น `NormalXOR.m`, `NoisyXOR.m`, `MNIST.m`
- สคริปต์ MATLAB จะบันทึกผลเป็นไฟล์ CSV ใน `MATLAB/MATLAB/result/...` ซึ่งมีคอลัมน์สำคัญ ได้แก่ `Accuracy_on_test_data`, `Accuracy_on_training_data`, `Time` (ใช้ผล Accuracy และค่า `Time` สำหรับ Training Time)

ตัวอย่างการรันแบบไม่มี GUI
- Windows (แนะนำ):
  - เปิด `CMD` หรือ `PowerShell`
  - รัน: `matlab -batch "NormalXOR"`
  - หรือกำหนดพารามิเตอร์เอง: `matlab -batch "NormalXOR(\"number_of_clauses\", 200, \"T\", 10, \"s\", 5.0, \"states\", 100, \"epochs\", 100)"`
- macOS/Linux:
  - รัน: `matlab -batch "NormalXOR"` (หากสคริปต์อ้างพาธ Windows สำหรับ PID ให้ปรับแก้ไฟล์ `.m` เรื่องพาธชั่วคราวตามระบบปฏิบัติการก่อน)

หมายเหตุเกี่ยวกับไฟล์ PID ชั่วคราว
- สคริปต์ `.m` มีการสร้างไฟล์ `C:\\temp\\training_pid.txt` แต่ในงานนี้ไม่ใช้สำหรับวัดทรัพยากร ให้พิจารณาเฉพาะไฟล์ผลลัพธ์ CSV ใน `MATLAB/MATLAB/result/...`

ไฟล์ผลลัพธ์ที่เกี่ยวข้อง (ตัวอย่าง)
- `MATLAB/MATLAB/result/normal_xor/normalXOR_result_log.csv`
- `MATLAB/MATLAB/result/noisy_xor/...csv`
- `MATLAB/MATLAB/result/mnist/...csv`

หมายเหตุ
- หากต้องการปรับให้ตรงสภาพแวดล้อม macOS/Linux ให้เปลี่ยนพาธไฟล์ PID ในสคริปต์ `.m` จาก `C:\\temp\\...` เป็นตำแหน่งชั่วคราวของระบบ เช่น `/tmp/training_pid.txt`

## 3.5.2 ทดสอบ Tsetlin Machine บน Python โดยใช้ค่าพารามิเตอร์เดียวกัน
- ใช้สคริปต์ใน `TsetlinMachine-purePython/` เช่น `NormalXORPure.py`, `NoisyXORPure.py`, `MNISTPure.py`
- ปรับค่าพารามิเตอร์ให้ “เหมือนกับ MATLAB” ได้แก่ `number_of_clauses`, `T`, `s`, `states`, `epochs` และโครงสร้างข้อมูลอินพุต/เอาต์พุต
- บันทึก Accuracy และ Training Time จากไฟล์ผลลัพธ์ของสคริปต์ Python เพื่อเปรียบเทียบ

ตัวอย่างการรัน
- ปกติ: `python TsetlinMachine-purePython/NormalXORPure.py`
- หากต้องการปรับพารามิเตอร์ ให้แก้ค่าที่ส่วนต้นไฟล์สคริปต์ (ตัวแปรพารามิเตอร์) ให้ตรงกับฝั่ง MATLAB

ไฟล์ผลลัพธ์ที่เกี่ยวข้อง (ตัวอย่าง)
- `TsetlinMachine-purePython/result/normal_xor_pure_python/normal_xor_pure_python_result_log.csv`
- `TsetlinMachine-purePython/result/noisy_xor_pure_python/...csv`

หมายเหตุ
- การบันทึกเวลา/Accuracy บน Python มีในสคริปต์อยู่แล้ว หากต้องการรูปแบบคอลัมน์ให้เหมือน MATLAB ให้ปรับชื่อคอลัมน์และลำดับก่อนนำไปวิเคราะห์รวม

## 3.5.3 วิเคราะห์และเปรียบเทียบผลลัพธ์
วิเคราะห์เชิงปริมาณด้วยตารางและกราฟเพื่อแสดงแนวโน้ม โดยเปรียบเทียบหัวข้อสำคัญดังนี้
- ความแม่นยำของโมเดล: Accuracy บนชุดทดสอบและชุดฝึก
- ประสิทธิภาพการประมวลผล: Training Time

แนวทางการสรุปผล
- รวมผลลัพธ์จากไฟล์ CSV ของ MATLAB และ Python เป็นตารางสรุปชุดเดียว (เลือกเฉพาะ Accuracy และ Training Time)
- สร้างกราฟเปรียบเทียบ (เช่น Bar/Line) สำหรับเมตริก: Accuracy, Training Time
- อธิบายความแตกต่างที่พบ โดยเชื่อมโยงกับสถาปัตยกรรมรันไทม์ (MATLAB vs Python) และลักษณะการอิมพลีเมนต์ของ Tsetlin Machine ในแต่ละภาษา
- สรุปผลว่าในเงื่อนไขเดียวกัน ฝั่งใดมีประสิทธิภาพ/การใช้ทรัพยากร/ความแม่นยำที่ดีกว่า พร้อมข้อเสนอแนะการใช้งานตามกรณี

## เช็กลิสต์ยืนยันความเท่าเทียมของการทดลอง
- ใช้พารามิเตอร์เดียวกันทุกตัว: `number_of_clauses`, `T`, `s`, `states`, `epochs`
- ใช้ชุดข้อมูลเดียวกัน และการเตรียมอินพุตในรูปแบบเดียวกัน
- เก็บสถิติที่จำเป็นเหมือนกัน: Accuracy และ Training Time
- บันทึกคอลัมน์ผลลัพธ์เป็นรูปแบบเดียวกันก่อนนำไปวิเคราะห์รวม

## ไฟล์/โฟลเดอร์ที่อ้างอิง
- `MATLAB/MNIST.m`, `MATLAB/NormalXOR.m`, `MATLAB/NoisyXOR.m`
- `MATLAB/TsetlinMachine.m`
- `MATLAB/MATLAB/result/...` เช่น `normal_xor/normalXOR_result_log.csv`, `mnist/mnist_result_log.csv`
- `TsetlinMachine-purePython/MNISTPure.py`, `TsetlinMachine-purePython/NormalXORPure.py`, `TsetlinMachine-purePython/NoisyXORPure.py`
- `TsetlinMachine-purePython/result/...` เช่น `normal_xor_pure_python/normal_xor_pure_python_result_log.csv`

## หมายเหตุเพิ่มเติม
- ใน repo นี้มีตัวอย่างไฟล์ CSV ผลลัพธ์ทั้งฝั่ง MATLAB และ Python แล้ว สามารถใช้เป็นแม่แบบในการสรุปเฉพาะ Accuracy และ Training Time ได้ทันที

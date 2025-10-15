# 3.5 วิธีการทดลอง: MATLAB vs Python (Tsetlin Machine)

เอกสารนี้สรุปขั้นตอนการทดลองตามหัวข้อ 3.5 โดยเน้นการทดสอบแบบไม่มี GUI, การบันทึกการใช้ทรัพยากรด้วย psutil, การบันทึก Accuracy/เวลา และแนวทางการวิเคราะห์เปรียบเทียบผลลัพธ์ระหว่าง MATLAB และ Python ให้ใช้พารามิเตอร์ชุดเดียวกันเพื่อความยุติธรรม

## 3.5.1 ทดสอบ Tsetlin Machine บน MATLAB (ไม่มี GUI) และการวัดผล
- รัน MATLAB แบบไม่มี GUI เพื่อลด overhead จากหน้าต่างโปรแกรม
- ใช้ไฟล์ `.m` ในโฟลเดอร์ `MATLAB/` เช่น `NormalXOR.m`, `NoisyXOR.m`, `MNIST.m`
- สคริปต์ MATLAB จะบันทึกผลเป็นไฟล์ CSV ใน `MATLAB/MATLAB/result/...` ซึ่งมีคอลัมน์สำคัญ ได้แก่ `Accuracy_on_test_data`, `Accuracy_on_training_data`, `Time`
- ใช้ psutil (ฝั่ง Python) เฝ้าดูการใช้ CPU/RAM ระหว่างการรัน โดยอ่าน PID จากไฟล์ที่สคริปต์ MATLAB สร้าง (`C:\\temp\\training_pid.txt`) แล้วติดตามจนกระทั่งจบงาน

ตัวอย่างการรันแบบไม่มี GUI
- Windows (แนะนำ):
  - เปิด `CMD` หรือ `PowerShell`
  - รัน: `matlab -batch "NormalXOR"`
  - หรือกำหนดพารามิเตอร์เอง: `matlab -batch "NormalXOR(\"number_of_clauses\", 200, \"T\", 10, \"s\", 5.0, \"states\", 100, \"epochs\", 100)"`
- macOS/Linux:
  - รัน: `matlab -batch "NormalXOR"` (หากสคริปต์อ้างพาธ Windows สำหรับ PID ให้ปรับแก้ไฟล์ `.m` เรื่องพาธชั่วคราวตามระบบปฏิบัติการก่อน)

การบันทึกสถิติ CPU/RAM ด้วย psutil (แนวทาง)
- ติดตั้ง: `pip install psutil`
- หลักการ: อ่าน PID จาก `C:\\temp\\training_pid.txt` แล้วใช้ psutil วัด `cpu_percent`, `memory_info().rss` เป็นช่วงๆ จนงานเสร็จ
- บันทึกลงไฟล์ CSV แยก เช่น `result/matlab_resource_log.csv` เพื่อใช้วิเคราะห์ร่วมกับผลลัพธ์จาก MATLAB

ไฟล์ผลลัพธ์ที่เกี่ยวข้อง (ตัวอย่าง)
- `MATLAB/MATLAB/result/normal_xor/normalXOR_result_log.csv`
- `MATLAB/MATLAB/result/noisy_xor/...csv`
- `MATLAB/MATLAB/result/mnist/...csv`

หมายเหตุ
- หากต้องการปรับให้ตรงสภาพแวดล้อม macOS/Linux ให้เปลี่ยนพาธไฟล์ PID ในสคริปต์ `.m` จาก `C:\\temp\\...` เป็นตำแหน่งชั่วคราวของระบบ เช่น `/tmp/training_pid.txt`

## 3.5.2 ทดสอบ Tsetlin Machine บน Python โดยใช้ค่าพารามิเตอร์เดียวกัน
- ใช้สคริปต์ใน `TsetlinMachine-purePython/` เช่น `NormalXORPure.py`, `NoisyXORPure.py`, `MNISTPure.py`
- ปรับค่าพารามิเตอร์ให้ “เหมือนกับ MATLAB” ได้แก่ `number_of_clauses`, `T`, `s`, `states`, `epochs` และโครงสร้างข้อมูลอินพุต/เอาต์พุต
- ใช้ psutil เฝ้าดู CPU/RAM ขณะรันสคริปต์ Python เช่นเดียวกับกรณี MATLAB
- บันทึก Accuracy และเวลา รวมทั้งบันทึกทรัพยากรลงไฟล์ CSV เพื่อเปรียบเทียบ

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
- ประสิทธิภาพการประมวลผล: Training Time, Prediction Time
- การใช้ทรัพยากรระบบ: ค่าเฉลี่ย/สูงสุดของ CPU Usage และ RAM Usage ตลอดการรัน
- ความแม่นยำของโมเดล: Accuracy บนชุดทดสอบและชุดฝึก

แนวทางการสรุปผล
- รวมผลลัพธ์จาก CSV ของ MATLAB และ Python และจากไฟล์ทรัพยากรที่วัดด้วย psutil เป็นตารางสรุปชุดเดียว
- สร้างกราฟเปรียบเทียบ (เช่น Bar/Line) สำหรับแต่ละเมตริก: เวลา, CPU, RAM, Accuracy
- อธิบายความแตกต่างที่พบ โดยเชื่อมโยงกับสถาปัตยกรรมรันไทม์ (MATLAB vs Python), การใช้หน่วยความจำ, และลักษณะการอิมพลีเมนต์ของ Tsetlin Machine ในแต่ละภาษา
- สรุปผลว่าในเงื่อนไขเดียวกัน ฝั่งใดมีประสิทธิภาพ/การใช้ทรัพยากร/ความแม่นยำที่ดีกว่า พร้อมข้อเสนอแนะการใช้งานตามกรณี

## เช็กลิสต์ยืนยันความเท่าเทียมของการทดลอง
- ใช้พารามิเตอร์เดียวกันทุกตัว: `number_of_clauses`, `T`, `s`, `states`, `epochs`
- ใช้ชุดข้อมูลเดียวกัน และการเตรียมอินพุตในรูปแบบเดียวกัน
- วัดเวลา/ทรัพยากรครอบคลุมช่วงเดียวกันของการรัน (เริ่ม-จบ)
- บันทึกคอลัมน์ผลลัพธ์เป็นรูปแบบเดียวกันก่อนนำไปวิเคราะห์รวม

## ไฟล์/โฟลเดอร์ที่อ้างอิง
- `MATLAB/MNIST.m`, `MATLAB/NormalXOR.m`, `MATLAB/NoisyXOR.m`
- `MATLAB/TsetlinMachine.m`
- `MATLAB/MATLAB/result/...`
- `TsetlinMachine-purePython/MNISTPure.py`, `TsetlinMachine-purePython/NormalXORPure.py`, `TsetlinMachine-purePython/NoisyXORPure.py`
- `TsetlinMachine-purePython/result/...`

## หมายเหตุเพิ่มเติม
- หากต้องการสคริปต์ Python สำหรับเฝ้าดู PID และบันทึก CPU/RAM ให้แจ้งได้ จะเพิ่มตัวอย่างโค้ด `monitor_resources.py` ที่อ่าน PID จากไฟล์และบันทึกเป็น CSV เพื่อใช้งานร่วมกันทั้งฝั่ง MATLAB และ Python

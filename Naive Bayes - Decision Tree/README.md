# Live Demo — Naive Bayes vs Decision Tree

Demo tương tác để thuyết trình trước lớp: chọn (hoặc tự chỉnh) một khách hàng ngân hàng
và **xem hai thuật toán "suy nghĩ" khác nhau thế nào** trên cùng một người.

## Chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```
Trình duyệt tự mở `http://localhost:8501`. Dữ liệu nằm sẵn ở `data/bank-full.csv`
(nếu thiếu, app sẽ cho tải file lên). Lần chạy đầu mất ~10 giây để huấn luyện.

**Mẹo chiếu lớp:** Ctrl + "+" để phóng to trình duyệt; đóng sidebar (nút « ) nếu cần thêm chỗ;
trỏ chuột vào cây quyết định → nút ⤢ để xem toàn màn hình.

## Cấu trúc
| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit (4 tab) |
| `demo_core.py` | Naive Bayes **viết tay** (~60 dòng, không hộp đen), cây quyết định, biểu đồ |
| `data/bank-full.csv` | UCI Bank Marketing, 45.211 khách hàng |

## Kịch bản trình bày (~8–10 phút)
1. **Mở đầu (tab 🎯).** Chọn khách ① (27 tuổi, quản lý, chưa từng được liên hệ). Cả hai mô hình cùng đáp
   "KHÔNG đăng ký" — *"cùng đáp án, nhưng đi bằng hai con đường khác hẳn nhau"*.
2. **Naive Bayes (tab 🧮).** Đọc "Bước 0": chỉ 11,7% khách đăng ký. Kéo thanh trượt từ 0 → 14, mỗi thông tin
   "bỏ phiếu" (×0,41 nghiêng KHÔNG, ×1,63 nghiêng ĐĂNG KÝ…), xác suất đổi theo. Nhấn mạnh: học = **đếm**.
3. **"Nhìn sâu"**: chọn *Kênh liên hệ* → đọc câu "trong 4.231 khách đã đăng ký có … người dùng cellular".
   Chọn *Thời lượng cuộc gọi* → chỉ cho thấy đường cong chuẩn NB giả định so với dữ liệu thật (cột mờ).
4. **Thí nghiệm "Naive"**: kéo xem xác suất NB tụt khi lặp lại 1 thông tin, cây (đường xanh) thì đứng yên
   → hiểu vì sao gọi là *ngây thơ* (giả định độc lập).
5. **Decision Tree (tab 🌳).** Đọc đường viền đỏ từ gốc tới lá, đọc 3 câu hỏi Có/Không. Kéo *độ sâu* 1→5 ở
   sidebar: cây lớn dần. Mở biểu đồ **Overfitting**: cây quá sâu học thuộc lòng.
6. **Bất đồng.** Chọn khách ④: NB nói ~88% ĐĂNG KÝ (đúng), cây nói ~3% (sai). Giải thích: NB gom nhiều bằng chứng
   nhỏ, còn cây rơi vào lá mà đa số là "không".
7. **Rò rỉ thông tin.** Bỏ tick *duration* ở sidebar: cả hai mô hình yếu đi rõ rệt (AUC ~0,87 → ~0,76 với NB).
   *"Thông tin mạnh nhất lại chỉ biết sau khi cuộc gọi kết thúc."*
8. **Ứng dụng (tab 📞).** Kéo "gọi bao nhiêu %": gọi 20% danh sách theo xếp hạng tìm được ~60–70% người đăng ký
   (gấp ~3 lần gọi ngẫu nhiên). Chỉ ra "bẫy Accuracy" (luôn đoán KHÔNG đã đạt 88,3%).

## Câu hỏi thầy cô có thể hỏi
- **Vì sao dùng P(đăng ký) chứ không chỉ Có/Không?** Vì tập lệch (11,7% "yes"); xếp hạng theo xác suất hữu ích hơn.
- **Naive Bayes có làm mịn không?** Có — Laplace (+1) cho feature phân loại, tránh xác suất 0.
- **Đường cong chuẩn có hợp lý không?** Không hoàn toàn (dữ liệu lệch phải) — chính là điểm "naive"; vẫn dùng được vì chỉ so sánh 2 đường cong.
- **Cây chọn câu hỏi thế nào?** Chọn ngưỡng giảm độ tạp (Gini) nhiều nhất ở mỗi nút.

## Ghi chú kỹ thuật (để trả lời cho trung thực)
- Demo dùng 14/16 feature (bỏ `day`, `pdays`) cho gọn. Chia train/test 80/20, `stratify`, `random_state=42` giống pipeline TV2.
- NB trong demo = đếm tần suất (phân loại) + phân phối chuẩn (số). Đã đối chiếu với
  `CategoricalNB + GaussianNB` của sklearn: sai khác log-odds < 1e-10. **Khác** với `GaussianNB` trên one-hot của TV3, nên số liệu sẽ không trùng hoàn toàn.
- Khách hàng ở 4 mẫu có sẵn đều là **khách thật trong tập test** (chưa được dùng để học).

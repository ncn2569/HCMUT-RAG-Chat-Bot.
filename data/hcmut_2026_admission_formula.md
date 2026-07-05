# Đặc tả Công thức Tính điểm Xét tuyển Tổng hợp HCMUT — Năm 2026

> Tài liệu này mô tả đầy đủ công thức, biến số đầu vào, thứ tự tính toán và các
> điểm cần lưu ý để một agent lập trình (Claude Code / Cursor / Copilot...) có
> thể hiện thực hóa hàm `predict_admission` đúng theo quy chế 2026 của HCMUT.
> Đây là bản cập nhật thay thế hoàn toàn công thức 75-20-5 đơn giản trước đó.

---

## 1. Công thức tổng quát

```
[Điểm Xét tuyển] = [Điểm học lực]⁽¹⁾ + [Điểm cộng]⁽²⁾ + [Điểm ưu tiên]⁽³⁾
```

- Thang điểm: **100**
- [Điểm Xét tuyển] tối đa = 100 (được ràng buộc bởi công thức Điểm cộng và Điểm ưu tiên bên dưới, không thể vượt quá 100 dù cộng dồn).

---

## 2. [Điểm học lực]⁽¹⁾ — Thang điểm 100

> **Quy tắc làm tròn:** làm tròn đến **0.01** ở TỪNG thành tố, và làm tròn
> đến **0.01** ở điểm tổng [Điểm học lực] sau khi cộng.

```
[Điểm học lực] = [Điểm năng lực] × 70%
              + [Điểm TNTHPT quy đổi] × 20%
              + [Điểm học THPT quy đổi] × 10%
```

Cách tính `[Điểm năng lực]`, `[Điểm TNTHPT quy đổi]`, `[Điểm học THPT quy đổi]`
**khác nhau tùy theo thí sinh có hay không có kết quả thi ĐGNL ĐHQG-HCM 2026.**

### 2.1. Đối tượng CÓ kết quả thi ĐGNL ĐHQG-HCM năm 2026

```
[Điểm năng lực] = [Điểm ĐGNL] / 15
    # Thang điểm ĐGNL gốc = 1500, quy đổi sang thang 100 bằng cách chia 15

[Điểm TNTHPT quy đổi] = [Tổng điểm thi 3 môn TNTHPT trong tổ hợp] / 4 × 10

[Điểm học THPT quy đổi] = [Trung bình cộng điểm TB lớp 10, 11, 12
                            của các môn trong tổ hợp] × 10
```

### 2.2. Đối tượng KHÔNG CÓ kết quả thi ĐGNL ĐHQG-HCM năm 2026

```
[Điểm năng lực] = [Điểm TNTHPT quy đổi] × 0.75

[Điểm TNTHPT quy đổi] = [Tổng điểm thi 3 môn TNTHPT trong tổ hợp] / 4 × 10
    # (công thức TNTHPT quy đổi giống hệt mục 2.1)

[Điểm học THPT quy đổi] = [Trung bình cộng điểm TB lớp 10, 11, 12
                            của các môn trong tổ hợp] × 10
    # (giống hệt mục 2.1)
```

> ⚠️ **Cần đối chiếu lại với nguồn chính thức trước khi triển khai production:**
> Ảnh gốc ghi "Tổng điểm thi **3 môn** TNTHPT... / **4** × 10". Về mặt số học,
> nếu tổ hợp thật sự chỉ có 3 môn (mỗi môn tối đa 10 → tổng tối đa 30), công
> thức `/4 × 10` cho ra tối đa 75/100 chứ không phải 100/100 — có khả năng đây
> là do kỳ thi TNTHPT từ 2025 có cấu trúc 4 môn (2 bắt buộc + 2 tự chọn) và tổ
> hợp xét tuyển thực tế dùng 4 môn (tổng tối đa 40 → `/4×10` = tối đa 100, hợp
> lý về mặt toán học). Đề xuất: xác nhận với phòng đào tạo HCMUT xem tổ hợp
> dùng **3 hay 4 môn** trước khi cứng hoá logic; nếu là 3 môn, số chia đúng có
> lẽ phải là `/3` chứ không phải `/4`.

---

## 3. [Điểm cộng]⁽²⁾ — Điểm thành tích đặc biệt

> Tổng điểm cộng, điểm thưởng, điểm xét thưởng, điểm khuyến khích (gọi chung
> là Điểm cộng) không vượt quá **10% mức điểm tối đa của thang điểm xét tuyển**
> (tối đa **10 điểm** trên thang 100).

```
[Điểm cộng thành tích] = [Điểm thưởng] + [Điểm xét thưởng] + [Điểm khuyến khích]

# Trường hợp 1:
Nếu [Điểm học lực] + [Điểm cộng thành tích] < 100:
    [Điểm cộng] = [Điểm cộng thành tích]

# Trường hợp 2:
Nếu [Điểm học lực] + [Điểm cộng thành tích] ≥ 100:
    [Điểm cộng] = 100 - [Điểm học lực]
```

Thành phần của `[Điểm cộng thành tích]`:

| Thành phần | Tối đa | Đối tượng áp dụng |
|---|---|---|
| **Điểm thưởng** | 10 điểm | Thí sinh thuộc diện xét tuyển thẳng theo quy chế Bộ GD&ĐT (đối tượng 1.2, 1.3) nhưng KHÔNG dùng quyền xét tuyển thẳng. Chỉ được cộng **một lần duy nhất**. |
| **Điểm xét thưởng** | 5 điểm | Thí sinh có thành tích học tập nổi bật (không thuộc diện Điểm thưởng), năng khiếu văn-thể-mỹ, hoạt động xã hội. |
| **Điểm khuyến khích** | 5 điểm | Thí sinh có chứng chỉ ngoại ngữ quốc tế khác tiếng Anh, hoặc chứng chỉ quốc tế khác (Tin học quốc tế: MOS, IC3...). |

> Lưu ý: Chứng chỉ Tiếng Anh **không** dùng ở đây — nó chỉ dùng để quy đổi vào
> `[Điểm TNTHPT quy đổi]` (xem Mục 5), không tính vào Điểm khuyến khích.

---

## 4. [Điểm ưu tiên]⁽³⁾ — Ưu tiên khu vực / đối tượng

```
[Điểm ưu tiên quy đổi] = [Điểm ưu tiên (khu vực, đối tượng) theo Bộ GD&ĐT] / 3 × 10
    # Điểm ưu tiên theo Bộ GD&ĐT: thang 30, tối đa 2.75
    # => Điểm ưu tiên quy đổi thang 100: tối đa 9.17

# Trường hợp 1:
Nếu [Điểm học lực] + [Điểm cộng] < 75:
    [Điểm ưu tiên] = [Điểm ưu tiên quy đổi]

# Trường hợp 2:
Nếu [Điểm học lực] + [Điểm cộng] ≥ 75:
    [Điểm ưu tiên] = (100 - [Điểm học lực] - [Điểm cộng]) / 25 × [Điểm ưu tiên quy đổi]
    # làm tròn đến 0.01
```

---

## 5. Quy đổi chứng chỉ Tiếng Anh quốc tế (áp dụng năm 2026)

> Áp dụng để quy đổi **điểm môn Tiếng Anh trong điểm thi TNTHPT**
> (KHÔNG áp dụng cho điểm học bạ — khác với công thức năm 2025 trước đó).

Điều kiện tối thiểu để được quy đổi: **IELTS Academic ≥ 6.0** / **TOEFL iBT ≥ 60**
/ **TOEIC Nghe-Đọc ≥ 570 & Nói-Viết ≥ 310** / **PTE Academic ≥ 47**.

| IELTS Academic | PTE Academic | TOEFL iBT | TOEIC Nghe & Đọc | TOEIC Nói & Viết | Điểm môn Tiếng Anh quy đổi |
|---|---|---|---|---|---|
| ≥ 8.0 | ≥ 79 | ≥ 110 | ≥ 905 | ≥ 390 | **10.0** |
| 7.5 | 71–78 | 102–109 | 835–900 | 380–389 | **9.5** |
| 7.0 | 63–70 | 94–101 | 785–830 | 360–379 | **9.0** |
| 6.5 | 55–62 | 79–93 | 685–780 | 330–359 | **8.5** |
| 6.0 | 47–54 | 60–78 | 570–680 | 310–329 | **8.0** |

**Ghi chú quan trọng về TOEIC:** để được quy đổi, thí sinh phải đạt **đồng
thời cả hai** cặp điểm thành phần Nghe-Đọc VÀ Nói-Viết ở cùng một mức (hàng)
trong bảng. Nếu chỉ một trong hai cặp đạt yêu cầu ở mức cao hơn, điểm quy đổi
cuối cùng sẽ lấy theo **cặp điểm thành phần thấp hơn** (tức là lấy giá trị
`min` giữa hàng ứng với Nghe-Đọc và hàng ứng với Nói-Viết).

Sau khi có điểm Tiếng Anh quy đổi (0–10), thay thế điểm môn Tiếng Anh gốc
trong tổng điểm 3 (hoặc 4, xem cảnh báo Mục 2) môn của tổ hợp thi TNTHPT
trước khi tính `[Điểm TNTHPT quy đổi]` ở Mục 2.

---

## 6. Biến số đầu vào cần thu thập từ người dùng

| Biến | Kiểu | Bắt buộc | Ghi chú |
|---|---|---|---|
| `major` | string | ✅ | Tên ngành xét tuyển, dùng để so với điểm chuẩn |
| `has_dgnl` | boolean | ✅ | Có tham dự kỳ thi ĐGNL ĐHQG-HCM 2026 hay không → quyết định dùng công thức 2.1 hay 2.2 |
| `dgnl_score` | float (0–1500) | Nếu `has_dgnl = true` | Điểm thi ĐGNL |
| `thpt_scores_by_subject` | list[float] (mỗi môn 0–10) | ✅ | Điểm từng môn trong tổ hợp thi TNTHPT (3 hoặc 4 môn — xem Mục 2) |
| `english_cert` | object hoặc null | optional | `{type, ielts?, pte?, toefl?, toeic_listen_read?, toeic_speak_write?}` — nếu có, dùng để thay điểm môn Tiếng Anh trong `thpt_scores_by_subject` |
| `hocba_avg_by_subject` | list[float] (mỗi môn 0–10) | ✅ | Trung bình cộng điểm TB lớp 10-11-12 của từng môn trong tổ hợp |
| `diem_thuong` | float (0–10) | optional, default 0 | |
| `diem_xet_thuong` | float (0–5) | optional, default 0 | |
| `diem_khuyen_khich` | float (0–5) | optional, default 0 | |
| `diem_uu_tien_bo_gd` | float (0–2.75) | optional, default 0 | Điểm ưu tiên khu vực/đối tượng theo thang 30 của Bộ GD&ĐT |

---

## 7. Thứ tự tính toán (pseudocode tham khảo)

```text
function tinh_diem_xet_tuyen(input):

    # --- Bước A: Quy đổi chứng chỉ Tiếng Anh (nếu có) ---
    if input.english_cert exists:
        diem_anh_quy_doi = tra_bang_quy_doi_tieng_anh_2026(input.english_cert)
        if diem_anh_quy_doi is not None:
            thay thế điểm môn Tiếng Anh trong input.thpt_scores_by_subject
            bằng diem_anh_quy_doi

    # --- Bước B: Điểm TNTHPT quy đổi & Điểm học THPT quy đổi (dùng chung cho cả 2 đối tượng) ---
    tong_diem_tnthpt = sum(input.thpt_scores_by_subject)
    diem_tnthpt_quy_doi = round(tong_diem_tnthpt / 4 * 10, 2)   # xem cảnh báo Mục 2 về số chia 3 hay 4

    tb_hocba = average(input.hocba_avg_by_subject)
    diem_hocba_quy_doi = round(tb_hocba * 10, 2)

    # --- Bước C: Điểm năng lực (rẽ nhánh theo has_dgnl) ---
    if input.has_dgnl:
        diem_nang_luc = round(input.dgnl_score / 15, 2)
    else:
        diem_nang_luc = round(diem_tnthpt_quy_doi * 0.75, 2)

    # --- Bước D: Điểm học lực ---
    diem_hoc_luc = round(
        diem_nang_luc * 0.70 +
        diem_tnthpt_quy_doi * 0.20 +
        diem_hocba_quy_doi * 0.10,
        2
    )

    # --- Bước E: Điểm cộng ---
    diem_cong_thanh_tich = input.diem_thuong + input.diem_xet_thuong + input.diem_khuyen_khich
    if diem_hoc_luc + diem_cong_thanh_tich < 100:
        diem_cong = diem_cong_thanh_tich
    else:
        diem_cong = 100 - diem_hoc_luc

    # --- Bước F: Điểm ưu tiên ---
    diem_uu_tien_quy_doi = input.diem_uu_tien_bo_gd / 3 * 10
    if diem_hoc_luc + diem_cong < 75:
        diem_uu_tien = diem_uu_tien_quy_doi
    else:
        diem_uu_tien = round(
            (100 - diem_hoc_luc - diem_cong) / 25 * diem_uu_tien_quy_doi,
            2
        )

    # --- Bước G: Tổng điểm xét tuyển ---
    diem_xet_tuyen = diem_hoc_luc + diem_cong + diem_uu_tien

    return {
        diem_nang_luc, diem_tnthpt_quy_doi, diem_hocba_quy_doi,
        diem_hoc_luc, diem_cong, diem_uu_tien, diem_xet_tuyen
    }


function tra_bang_quy_doi_tieng_anh_2026(cert):
    # Bảng ở Mục 5. Trả về None nếu không đạt ngưỡng tối thiểu (IELTS 6.0 tương đương).
    # Với TOEIC: tra riêng hàng theo Nghe-Đọc và hàng theo Nói-Viết,
    # kết quả cuối = giá trị thấp hơn giữa hai hàng.
    ...
```

---

## 8. So sánh với điểm chuẩn & lời khuyên (giữ nguyên logic cũ, không đổi)

Sau khi có `diem_xet_tuyen`, so sánh với điểm chuẩn ngành `major` (dữ liệu ở
`data/admission_scores.json`, thang 100):

- Cao hơn điểm chuẩn > **2 điểm** → **"An toàn"**
- Chênh lệch trong khoảng **±1 điểm** → **"Cạnh tranh"**
- Thấp hơn điểm chuẩn → **"Nguy hiểm"**

Với thí sinh chưa đủ dữ liệu đầu vào (ví dụ chưa có điểm TNTHPT vì chưa thi),
áp dụng logic cũ: giả định thí sinh cần đạt cùng một tỉ lệ % hoàn thành ở các
thành phần còn thiếu để chạm điểm chuẩn, rồi báo số điểm thô cần đạt.

---

## 9. Danh sách việc cần xác nhận lại trước khi lên production

1. ❗ **Số chia trong `[Điểm TNTHPT quy đổi]`**: ảnh gốc ghi "3 môn... / 4",
   không khớp về mặt toán học nếu tổ hợp thật sự chỉ có 3 môn. Cần xác nhận tổ
   hợp xét tuyển 2026 dùng 3 hay 4 môn để chọn đúng số chia (3 hoặc 4).
2. ❗ Xác nhận thang điểm ĐGNL 2026 chính thức là **1500** (công thức chia 15)
   — khác với thang 1200 dùng ở các tài liệu/năm trước đó.
3. ❗ Với chứng chỉ Tiếng Anh, xác nhận cách xử lý khi thí sinh có chứng chỉ
   nhưng tổ hợp xét tuyển của họ **không dùng môn Tiếng Anh** (trường hợp này
   bảng quy đổi không áp dụng).
4. Xác nhận cách làm tròn: tài liệu ghi rõ làm tròn 0.01 ở [Điểm học lực], còn
   [Điểm ưu tiên] cũng ghi rõ làm tròn 0.01 — nhưng [Điểm cộng] và
   [Điểm Xét tuyển] tổng không có hướng dẫn làm tròn tường minh; nên áp dụng
   làm tròn 0.01 nhất quán cho toàn bộ hay giữ nguyên phần thập phân?

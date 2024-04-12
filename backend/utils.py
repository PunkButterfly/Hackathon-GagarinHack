ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_res(img_result):
    mapping = {
        'Drivers': ('driver_license', 1),
        'Drivers-2': ('driver_license', 2),
        'PTS': ('vehicle_passport', 0),
        'Passports': ('personal_passport', 1),
        'Passports-2': ('personal_passport', 2),
        'STS': ('vehicle_certificate', 1),
        'STS-2': ('vehicle_certificate', 2)
    }

    max_val = -1000
    doc_type = None

    for k,v in img_result.items():
        if max_val < v:
            max_val = v
            doc_type = mapping[k]

    return max_val, doc_type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# with open(f'{workdir}CH_CONFIG.yaml', 'r')  as f:
#     config = yaml.safe_load(f)

# CH_HOST = config['HOST']
# CH_PORT = config['PORT']
# CH_USERNAME = config['USER']
# CH_PASSWORD = config['PASSWORD']

def save_to_db(binary_data, file):
    # client = clickhouse_connect.get_client(host=CH_HOST, port=CH_PORT, username=CH_USERNAME, password=CH_PASSWORD)

    # s_bin = str(binary_data)[2:-1]

    # client.command("INSERT INTO GagarinHack2024.queries (bindata, filename) VALUES ('{}', '{}')".format(s_bin, file.filename))
    pass

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_cls_res(img_result):
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

def format_response_detect_client_prod(classifier_probs, recognited_text, predict_img_path, type='default'):
    parsed_cls_result = parse_cls_res(classifier_probs)

    ser = 'not recognized'
    ser_conf = -1.0

    numb = 'not recognized'
    numb_conf = -1.0

    for field_text, field_name, field_conf in recognited_text:
        if "ser" in field_name and ser_conf < field_conf:
            ser = field_text
            ser_conf = field_conf

        if "num" in field_name and numb_conf < field_conf:
            numb = field_text
            numb_conf = field_conf

    response = {
        "type": parsed_cls_result[1][0],
        "confidence": str(parsed_cls_result[0]),
        "series": ser,
        "number": numb,    
        "page_number": str(parsed_cls_result[1][1])
    }

    if type == 'punk_client':
        response["proceed_image_name"] = predict_img_path
        response["recognited_text"] =  recognited_text

    return response
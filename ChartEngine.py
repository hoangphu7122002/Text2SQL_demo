import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoProcessor
from util import *

def load_chart2table_model():
    model = Pix2StructForConditionalGeneration.from_pretrained("TeeA/ViDEPLOT-80").to('cuda')
    processor = AutoProcessor.from_pretrained("TeeA/ViDEPLOT-80")
    return model, processor

class SingletonMetaChart(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ChartInstanceModel(metaclass=SingletonMetaChart):
    def __init__(self):
        self.model,self.processor = load_chart2table_model()
    
    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor

class chartEngine:
    def __init__(self):
        self.model = ChartInstanceModel().get_model()
        self.tokenizer = ChartInstanceModel().get_processor()
    
    def inference(self,image):
        inputs = self.tokenizer(images=image, text="Generate underlying data table of the figure below:", truncation=True,
                        padding='max_length', max_length=512, return_tensors="pt", add_special_tokens=True, max_patches=2048).to('cuda')

        generated_ids = self.model.generate(**inputs, max_new_tokens=1000)
        generated_caption = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_caption[0]  

    def extract_db_schema_from_table(self,table: str) -> pd.DataFrame:
        db_id = generate_db_name()
        table_name = generate_table_name()
        
        # Split the table into rows
        rows = table.split("<0x0A>")
        
        # Extract the header
        header = rows[0].split(" | ")
        
        # Initialize an empty list to store the data
        data = []
        inserts = []
        # Loop over the remaining rows and split them into cells
        for row in rows[1:]:
            cells = row.split(" | ")
            new_data = []
            for cell in cells:
                if isinteger(cell):
                    new_data.append(int(cell))
                elif isfloat(cell):
                    new_data.append(float(cell))
                else:
                    new_data.append(cell)
            data.append(new_data)
            inserts.append(f"""INSERT INTO {table_name} VALUES({', '.join([str(x) if not isinstance(x, str) else f"'{x}'" for x in new_data])});""")
    
        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=header)
        schema = generate_create_table_sql({
            "table_names": [table_name],
            "column_indices": [0] * len(header),
            "column_names": header,
            "column_types": map_dtypes(df.dtypes.tolist())
        })
        return {
            "schema": [schema['schema']],
            "value": inserts,
            "db_id": db_id,
        }
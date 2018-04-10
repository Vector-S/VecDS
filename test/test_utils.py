from veclib.utils import *




def test_load_save_json():
    cfg = {
        'para1' : '123',
        'para2' : '34'

    }
    save_json(cfg,'../tmp/sample.cfg')
    assert  cfg == load_json('../tmp/sample.cfg')





if __name__=='__main__':
    test_load_save_json()
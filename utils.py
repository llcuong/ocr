import os
import settings


def save_upload_image(fileObj):
    filename = fileObj.filename    
    name , ext = filename.split('.')    
    save_filename = 'upload.jpg'
    upload_image_path = settings.join_path(settings.SAVE_DIR, 
                                           save_filename)
    
    fileObj.save(upload_image_path)
    
    return upload_image_path


from googleapiclient.discovery import build
from google.oauth2 import service_account

def upload_file() :
  creds = service_account.Credentials.from_service_account_file('path/to/your/json/key.json', scopes=['https://www.googleapis.com/auth/drive'])
  drive_service = build('drive', 'v3', credentials=creds)
  file_metadata = {'name': 'MyFile.txt',  'parents': ['<folder_id>'] }  # ID of the folder where you want to upload
  file_path = 'path/to/your/local/file.txt'
  media = MediaFileUpload(file_path, mimetype='text/plain')
  file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def download_file(file_id, file_path) :
  # file_id = 'file_id_to_download'
  # file_path = 'path/to/save/downloaded/file.txt'

  request = drive_service.files().get_media(fileId=file_id)
  fh = io.FileIO(file_path, mode='wb')
  downloader = MediaIoBaseDownload(fh, request)
  done = False
  
  while not done:
    status, done = downloader.next_chunk()

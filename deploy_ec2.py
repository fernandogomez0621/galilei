"""
deploy_ec2.py — Script de despliegue automático en EC2 con boto3.
La instancia ya tiene los permisos IAM configurados, así que no necesitamos
Access Keys explícitas — boto3 usará el Instance Profile automáticamente.

Uso:
    python deploy_ec2.py

Requisitos previos:
    - La EC2 ya tiene un Instance Profile con permisos EC2 + SSM (o SSH access)
    - Streamlit app en app.py y requirements.txt en el mismo directorio
    - Puerto 8501 abierto en el Security Group de la instancia
"""

import boto3
import time
import os
import subprocess

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estos valores
# ─────────────────────────────────────────────
REGION = "us-east-1"                       # Región de tu EC2
INSTANCE_ID = "i-XXXXXXXXXXXXXXXXX"       # ID de tu instancia EC2
APP_DIR = "/home/ec2-user/galilei"         # Directorio en la EC2
LOCAL_FILES = ["app.py", "requirements.txt"]  # Archivos a subir

# ─────────────────────────────────────────────
# CLIENTE EC2 (usa el Instance Profile — no necesita keys)
# ─────────────────────────────────────────────
ec2 = boto3.client("ec2", region_name=REGION)
ssm = boto3.client("ssm", region_name=REGION)

def get_instance_public_ip(instance_id: str) -> str:
    """Obtiene la IP pública de la instancia."""
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    ip = resp["Reservations"][0]["Instances"][0].get("PublicIpAddress", "")
    print(f"[INFO] IP pública de la instancia: {ip}")
    return ip

def run_ssm_command(instance_id: str, commands: list[str]) -> str:
    """
    Ejecuta comandos en la EC2 vía SSM Send Command.
    Alternativa a SSH — no requiere key pair ni puerto 22 abierto.
    La instancia necesita el agente SSM instalado (viene por defecto en Amazon Linux 2).
    """
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands},
        TimeoutSeconds=300
    )
    command_id = resp["Command"]["CommandId"]
    print(f"[INFO] SSM Command enviado: {command_id}")
    
    # Esperar resultado
    for _ in range(30):
        time.sleep(5)
        result = ssm.get_command_invocation(
            CommandId=command_id,
            InstanceId=instance_id
        )
        status = result["Status"]
        print(f"[INFO] Estado: {status}")
        if status in ["Success", "Failed", "TimedOut", "Cancelled"]:
            output = result.get("StandardOutputContent", "")
            error = result.get("StandardErrorContent", "")
            if error:
                print(f"[STDERR] {error}")
            return output
    
    return "Timeout esperando resultado del comando"

def upload_files_via_s3(files: list[str], bucket: str, prefix: str = "galilei/"):
    """
    Sube archivos a S3 y luego los descarga desde la EC2.
    Útil si no tienes SSH directo.
    """
    s3 = boto3.client("s3", region_name=REGION)
    s3_paths = []
    for f in files:
        if os.path.exists(f):
            key = f"{prefix}{os.path.basename(f)}"
            s3.upload_file(f, bucket, key)
            s3_paths.append(f"s3://{bucket}/{key}")
            print(f"[S3] Subido: {f} → s3://{bucket}/{key}")
    return s3_paths

def upload_files_via_scp(files: list[str], ip: str, key_path: str, remote_dir: str):
    """
    Alternativa: subir archivos vía SCP si tienes acceso SSH.
    """
    for f in files:
        cmd = f"scp -i {key_path} -o StrictHostKeyChecking=no {f} ec2-user@{ip}:{remote_dir}/"
        print(f"[SCP] {cmd}")
        subprocess.run(cmd, shell=True, check=True)

def setup_and_run_streamlit(instance_id: str, app_dir: str, s3_paths: list[str] = None):
    """
    Instala dependencias y lanza Streamlit como proceso en background.
    """
    download_cmds = []
    if s3_paths:
        download_cmds = [f"aws s3 cp {p} {app_dir}/ --region {REGION}" for p in s3_paths]
    
    commands = [
        f"mkdir -p {app_dir}",
        
        # Descargar archivos desde S3 si se usó esa estrategia
        *download_cmds,
        
        # Instalar Python y dependencias
        "which python3 || sudo yum install -y python3",
        "which pip3 || sudo yum install -y python3-pip",
        f"cd {app_dir} && pip3 install -r requirements.txt --quiet",
        
        # Matar instancia previa de Streamlit si existe
        "pkill -f 'streamlit run' || true",
        
        # Lanzar Streamlit en background con nohup
        f"cd {app_dir} && nohup python3 -m streamlit run app.py "
        f"--server.port 8501 --server.address 0.0.0.0 "
        f"--server.headless true "
        f"> {app_dir}/streamlit.log 2>&1 &",
        
        "sleep 3",
        "echo 'Streamlit iniciado'",
        f"tail -5 {app_dir}/streamlit.log"
    ]
    
    output = run_ssm_command(instance_id, commands)
    return output

def main():
    print("=" * 60)
    print("DEPLOY — Galilei Streamlit App en EC2")
    print("=" * 60)
    
    # Obtener IP de la instancia
    ip = get_instance_public_ip(INSTANCE_ID)
    
    # Opción 1: Subir via S3 (recomendado — no requiere SSH)
    # s3_paths = upload_files_via_s3(LOCAL_FILES, bucket="tu-bucket-galilei")
    # output = setup_and_run_streamlit(INSTANCE_ID, APP_DIR, s3_paths=s3_paths)
    
    # Opción 2: Subir via SCP (requiere SSH y key pair)
    # KEY_PATH = "~/.ssh/tu-key.pem"
    # upload_files_via_scp(LOCAL_FILES, ip, KEY_PATH, APP_DIR)
    # output = setup_and_run_streamlit(INSTANCE_ID, APP_DIR)
    
    # Opción 3: Solo lanzar (si los archivos ya están en la EC2)
    output = setup_and_run_streamlit(INSTANCE_ID, APP_DIR)
    
    print("\n[OUTPUT]", output)
    print("\n" + "=" * 60)
    print(f"✅ App disponible en: http://{ip}:8501")
    print("=" * 60)

if __name__ == "__main__":
    main()

import os
import glob
import subprocess
import time

# Configuração dos padrões de entrada e saída
INPUT_FILE_PATTERN = "QCD_*.txt"  # Padrão de arquivos de entrada
RENAMED_FILE_PATTERN_PREFIX = "QCD1_"  # Prefixo para arquivos renomeados

def process_file_with_script(input_file, script_name):
    """
    Executa o script de processamento passando o arquivo como argumento.
    Mostra as mensagens do script chamado diretamente no terminal.
    """
    print(f"Executando {script_name} para processar {input_file}...")
    result = subprocess.run(["python", script_name], text=True)
    if result.returncode == 0:
        print(f"Arquivo {input_file} processado com sucesso!")
        return True
    else:
        print(f"Erro ao processar {input_file}. Código de retorno: {result.returncode}")
        return False

def rewrite_content_and_rename(input_file, renamed_file):
    """
    Reescreve o conteúdo do arquivo e renomeia-o.
    """
    print(f"Reescrevendo o conteúdo do arquivo: {input_file}")
    
    # Novo conteúdo a ser inserido no arquivo
    new_content = """\

"""
    # Reescreve o conteúdo do arquivo
    with open(input_file, "w") as file:
        file.write(new_content)
    
    # Renomeia o arquivo
    os.rename(input_file, renamed_file)
    print(f"Arquivo renomeado para {renamed_file}")
    
    return renamed_file  # Retorna o nome do arquivo renomeado

def monitor_directory(script_name, input_pattern, renamed_prefix):
    """
    Monitora a pasta por arquivos não processados e os processa.
    """
    while True:
        # Listar arquivos que correspondem ao padrão inicial
        input_files = glob.glob(input_pattern)
        if not input_files:
            print("Nenhum arquivo novo encontrado. Aguardando...")
            time.sleep(10)  # Aguarda 10 segundos antes de verificar novamente
            continue
        
        for input_file in input_files:
            # Gerar o nome do arquivo renomeado explicitamente
            renamed_file = os.path.join(
                os.path.dirname(input_file),  # Diretório do arquivo original
                renamed_prefix + os.path.basename(input_file).split("_", 1)[1]  # Substitui o prefixo
            )
            
            print(f"Iniciando processamento do arquivo: {input_file}")
            
            # Processar o arquivo com o script específico
            if process_file_with_script(input_file, script_name):
                # Reescrever o conteúdo e renomear o arquivo
                new_file = rewrite_content_and_rename(input_file, renamed_file)
                
                # Processar o arquivo renomeado
                print(f"Processando novamente o arquivo renomeado: {new_file}")
                process_file_with_script(new_file, script_name)
            else:
                print(f"Erro ao processar {input_file}. Pulando para o próximo.")
        
        print("Verificação de novos arquivos concluída.")

if __name__ == "__main__":
    # Nome do script a ser chamado para processar os arquivos
    SCRIPT_NAME = "Event_Selection_QCD_region.py"
    
    # Iniciar o monitoramento do diretório com os padrões definidos
    monitor_directory(SCRIPT_NAME, INPUT_FILE_PATTERN, RENAMED_FILE_PATTERN_PREFIX)


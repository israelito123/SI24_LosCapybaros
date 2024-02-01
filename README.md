# Sistemas inteligentes

## Prerequisitos
Tener instalado:
- [anaconda](https://www.anaconda.com/) (Asegúrense de marcar la casilla de agregar al path)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

## Instrucciones
1. Haz un fork de este repositorio. Define el nombre como SI24_[nombre_de_equipo]
![create new fork](./imgs/forking.png)
2. Ve a tu nuevo repositorio y copia el link en code->Local-> HTTPS
![create new fork](./imgs/clone.png)
3. Clona tu forked repo localmente
```
git clone [http link]
```
4. Agrega el repositorio original como upstream

```
git remote add upstream https://github.com/JessicaBorja/SI24.git
```
5. Crea un ambiente e instala las dependencias adecuadas corriendo los siguientes comandos
```
conda create -n ml_env python=3.8
conda activate ml_env
pip install -e .
```
## Actualizar tu repositorio
Para actualizar tu repositorio local con los cambios del remoto ejecuta los siguientes comandos.

### Sincronizar repo local con remoto
### (Local -> Remoto)
Para mandar a github los commits que tengas localmente que no estén en el repositorio remoto, escribe en tu consola de comandos dentro de tu repositorio local:

```
git push
```

### (Remoto -> local)
Para traer a tu repositorio local los commits que se encuentran en el repositorio remoto, escribe en tu consola de comandos dentro de tu repositorio local:

```
git pull
```

### Sincronizar repo local y remoto con el upstream (original)
Los siguientes comandos:

1. (fetch) Buscan los cambios en el upstream
2. (merge) Integran los cambios del upstream a tu repositorio local
3. (push) Sincronizan los cambios del repositorio local (que vienen del upstream) a tu repositorio remoto en github

```
git fetch upstream
git merge upstream/master
git push
```

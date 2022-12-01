import tkinter as tk
ventana = tk.Tk()
ventana.title("Resultado")
ventana.geometry("500x100")
ventana.resizable(0,0)

ventana.configure(bg="cyan")
alerta = "Analisis de Discriminante Lineal"

cabezera = tk.Label(ventana,text = "                   El metodo mas eficaz es el Metodo de :                ",bg="cyan",fg="black")
msg = tk.Label(ventana,text = alerta,bg="cyan",fg="red")
cabezera.place(x=100,y=25)
msg.place(x=170,y=55)




import tkinter.messagebox
tkinter.messagebox.showinfo("Resultado","El metodo mas eficaz es el Metodo de :"+alerta)
ventana.mainloop()
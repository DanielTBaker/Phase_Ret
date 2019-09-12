import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import argparse
import astropy.units as u
from astropy.time import Time
import pickle as pkl
import sys
from matplotlib.colors import LogNorm
from emcee.utils import MPIPool
from mpi4py import MPI
from scipy.interpolate import interp1d

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ts = MPI.Wtime()

def svd_model(arr, nmodes=1):
    """                                                                                                                          
   Take time/freq visibilities SVD, zero out all but the largest                                                                
   mode, multiply original data by complex conjugate                                                                            

   Parameters                                                                                                                   
   ----------                                                                                                                   
   arr : array_like                                                                                                             
      Time/freq visiblity matrix                                                                                                

   Returns                                                                                                                      
   -------                                                                                                                      
   Original data array multiplied by the largest SVD mode conjugate                                                             
   """

    u, s, w = np.linalg.svd(arr)
    s[nmodes:] = 0.0
    S = np.zeros([len(u), len(w)], np.complex128)
    S[:len(s), :len(s)] = np.diag(s)

    model = np.dot(np.dot(u, S), w)
    return model

def Comp_plotter(arr1,
	         x_start=0,
	         x_end=1,
	         y_start=0,
	         y_end=1,
	         x_name='',
	         y_name='',
	         title='',
	         xlim=None,
	         ylim=None,
	         zlim=((None, None), (None, None), (None, None)),
	         rbinx=1,
	         rbiny=1):
	arr = np.mean(
	np.reshape(
		arr1,
		(arr1.shape[0] // rbiny, rbiny, arr1.shape[1] // rbinx, rbinx)),
		axis=(3, 1))
	fig, axes = plt.subplots(
	nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))
	if type(x_start) == u.quantity.Quantity:
		x_unit = '(%s)' % x_start.unit
		x_start = x_start.value
		x_end = x_end.value
	else:
		x_unit = ''
	if type(y_start) == u.quantity.Quantity:
		y_unit = '(%s)' % y_start.unit
		y_start = y_start.value
		y_end = y_end.value
	else:
		y_unit = ''

	ext = [x_start, x_end, y_end, y_start]
	im00 = axes[0, 0].imshow(
		np.imag(arr),
		aspect='auto',
		extent=ext,
		vmin=zlim[0][0],
		vmax=zlim[0][1])
	axes[0, 0].set_title('Imag')
	plt.colorbar(im00, ax=axes[0, 0])

	im01 = axes[0, 1].imshow(
		np.real(arr),
		aspect='auto',
		extent=ext,
		vmin=zlim[1][0],
		vmax=zlim[1][1])
	axes[0, 1].set_title('Real')
	plt.colorbar(im01, ax=axes[0, 1])

	im10 = axes[1, 0].imshow(
		np.abs(arr),
		aspect='auto',
		extent=ext,
		norm=LogNorm(),
		vmin=zlim[2][0],
		vmax=zlim[2][1])
	axes[1, 0].set_title('Abs')
	plt.colorbar(im10, ax=axes[1, 0])

	im11 = axes[1, 1].imshow(
		np.angle(arr),
		aspect='auto',
		extent=ext,
		cmap='twilight',
		vmin=-np.pi,
		vmax=np.pi)
	axes[1, 1].set_title('Phase')
	plt.colorbar(im11, ax=axes[1, 1])
	if not xlim == None:
		axes[1, 1].set_xlim(xlim)
	if not ylim == None:
		axes[1, 1].set_ylim(ylim)
	axes[0, 0].set_ylabel('%s %s' % (y_name, y_unit))
	axes[1, 0].set_ylabel('%s %s' % (y_name, y_unit))
	axes[1, 0].set_xlabel('%s %s' % (x_name, x_unit))
	axes[1, 1].set_xlabel('%s %s' % (x_name, x_unit))

	fig.suptitle(title, fontsize=16)

parser = argparse.ArgumentParser(description='Phase Retrieval Code')

parser.add_argument('-th', type=int,default = 80,help='Number of Threads')
parser.add_argument('-ds', type=str,default = 'GB057',help='Data Name')
parser.add_argument('-out', type=str,default = './',help='Save Directory')
parser.add_argument('-t1', type=str,default = 'ar',help='Telescope 1')
parser.add_argument('-t2', type=str,default = 'gb',help='Telescope 2')
parser.add_argument('-bff', type=int,default = 128,help='Number of Frequency Chunks')
parser.add_argument('-bft', type=int,default = 22,help='Number of Time Chunks')
parser.add_argument('-svd',action='store_false',default=True,help='Add this if the Data has already had (or does not need) an SVD')
parser.add_argument('-T',action='store_false',default=True,help='Add this if Data is already in the form [freq,time]')
parser.add_argument('-s1', action='store_true',default = False, help='Skip 1st Visibility Filter')
parser.add_argument('-si', action='store_true',default = False, help='Skip Intensity Filter')
parser.add_argument('-f',type=float,default=np.inf,help='Max Frequency')

args=parser.parse_args()
dirname=args.out
dset=args.ds
bf_t = args.bft
bf_f = args.bff
t1=args.t1
t2=args.t2

specs='%s_%s%s_%s_%s' %(dset,t1,t2,bf_f,bf_t)

if rank>0:
	##Load Files
	if rank==1:
		print('Importing Files',flush=True)
	GBAR_F=np.load('/home/p/pen/bakerd11/%s/dynamic_spectrum_%s%s.npz' %(dset,t2,t1))
	ARAR_F=np.load('/home/p/pen/bakerd11/%s/dynamic_spectrum_%s%s.npz' %(dset,t1,t1))
	GBGB_F=np.load('/home/p/pen/bakerd11/%s/dynamic_spectrum_%s%s.npz' %(dset,t2,t2))

	if args.T:
		GBAR=np.nan_to_num(GBAR_F['I'][:,GBAR_F['f_MHz']<=args.f].T)
		ARAR=ARAR_F['I'][:,ARAR_F['f_MHz']<=args.f].T
		GBGB=GBGB_F['I'][:,GBGB_F['f_MHz']<=args.f].T
	else:
		GBAR=np.nan_to_num(GBAR_F['I'][GBAR_F['f_MHz']<=args.f,:])
		ARAR=ARAR_F['I'][ARAR_F['f_MHz']<=args.f,:]
		GBGB=GBGB_F['I'][GBGB_F['f_MHz']<=args.f,:]

	##Data Shape
	nt = ARAR.shape[1]
	nf = ARAR.shape[0]
	##Chunk Size
	nt2 = nt//bf_t
	nf2 = nf//bf_f

	##Coordinates
	freqs=GBAR_F['f_MHz'][GBAR_F['f_MHz']<args.f]*u.MHz
	times=GBAR_F['t_s']*u.s

	##Determine Mask 
	std0=np.std(ARAR,axis=0)
	std0[std0>.5]=1
	std0[std0<1]=0
	starts=list()
	starts.append(0)
	stops=list()
	for i in range(std0.shape[0]-1):
		if std0[i+1]==1 and std0[i]==0:
			starts.append(i+1)
		if std0[i+1]==0 and std0[i]==1:
			stops.append(i+1)
	stops.append(std0.shape[0])

	Vmsk=np.zeros(std0.shape[0])
	for i in range(len(starts)):
		if stops[i]>starts[i]+20:
			Vmsk[starts[i]:stops[i]]=1

	ARAR*=Vmsk
	GBGB*=Vmsk
	GBAR*=Vmsk
	if rank==1:
		np.save('%s/DB_b0834_%s%s_Vmsk_%s.npy' % (dirname,t1,t1,dset),Vmsk)

	GBAR[:,Vmsk==0]=GBAR[:,Vmsk==1].mean()
	##Run SVD on Visibility if requested
	if args.svd:
		if rank==1:
			print('Performing SVD on Visibility',flush=True)
		GBAR_svd=svd_model(GBAR, nmodes=2)

		# GBAR*=np.exp(-1j*np.angle(GBAR_svd))
		# GBAR*=Vmsk

		GBAR/=GBAR_svd

	##Setup pyFFTW
	if rank==1:
		print('Preparing FFTs',flush=True)
	try:
		pyfftw.import_wisdom(
			pkl.load(open('pyfftwwis_niagara-%s.pkl' %dset,'rb')))
	except:
		if rank==1:
			print('No Wisdom Loaded',flush=True)

	##Prepare FFTs
	nthread = args.th
	fft_1 = pyfftw.empty_aligned((nf, nt),dtype='complex128')
	fft_2 = pyfftw.empty_aligned((nf, nt),dtype='complex128')
	fft_object_F = pyfftw.FFTW(
		fft_1, fft_2, axes=(0, 1), direction='FFTW_FORWARD', threads=nthread)
	fft_object_B = pyfftw.FFTW(
		fft_2, fft_1, axes=(0, 1), direction='FFTW_BACKWARD', threads=nthread)
	fft_object_F1 = pyfftw.FFTW(
		fft_1, fft_2, axes=(1,), direction='FFTW_FORWARD', threads=nthread)
	fft_object_B1 = pyfftw.FFTW(
		fft_2, fft_1, axes=(1,), direction='FFTW_BACKWARD', threads=nthread)

	pkl.dump(pyfftw.export_wisdom(),open('pyfftwwis_niagara-%s.pkl' %dset,'wb'))

	##Cordinates of secondary spectra
	f_d = np.fft.fftfreq(nt, d=times[1]).to(u.mHz)
	tau = np.fft.fftfreq(nf, d=freqs[1] - freqs[0]).to(u.ms)

	f_d_red = np.fft.fftfreq(nt2, d=times[1]).to(u.mHz)
	tau_red = np.fft.fftfreq(nf2, d=freqs[1] - freqs[0]).to(u.ms)


	ARAR_max=10
	ARAR_min=.5
	C_ARAR_max=1e-6
	C_ARAR_min=1e-12
	B1=4500
	B2=600
	f_arc_max=30
	tau_arc_max=.4

	##Plot Raw Arecibo Dspec 
	plt.figure()
	plt.imshow(
		ARAR,
		aspect='auto',
		vmax=ARAR_max,
		vmin=ARAR_min,
		extent=[times[0].value, times[-1].value, freqs[-1].value, freqs[0].value])
	plt.colorbar()
	plt.xlabel('Time (s)')
	plt.ylabel('Freq (MHz)')
	plt.title('%s%s Dynamic Spectrum' %(t1,t1))
	plt.ylim((freqs[0].value, freqs[-1].value))
	plt.savefig('%s/DB_b0834_%s%s_dspec_%s.png' % (dirname,t1,t1,dset))
	if rank==1:
		print('Filtering Dynamic Spectrum',flush=True)
		comm.send((bf_f,bf_t,nf,nt,nf2,nt2,freqs,times),dest=0,tag=1)

if rank==0:
	GBAR=None
	Smat=None
	N=None
	bf_t=0
	bf_f=0
	nf=0
	nt=0
	nf2=0
	nt2=0
	bf_f,bf_t,nf,nt,nf2,nt2,freqs,times = comm.recv(source=1,tag=1)
	Vmsk=np.load('%s/DB_b0834_%s%s_Vmsk_%s.npy' % (dirname,t1,t1,dset))
	print('Number of workers: %s' %(size-1),flush=True)
	tasks=list()
	for i in range(bf_f):
		for j in range(bf_t):
			tasks.append((i,j))
comm.Barrier()

if not args.si:
	if rank>0:
		# temp=ARAR[:,Vmsk==1].mean()
		temp = 0
		ARAR -= temp
		ARAR *= Vmsk

		##ARAR Power Spectrum
		fft_1[:] = np.copy(ARAR)
		fft_object_F()
		C = np.abs(fft_2 / (nf * nt))**2
		fft_2[:] = np.copy(C)
		fft_object_B()
		fft_1 /= np.fft.ifft(np.abs(np.fft.fft(Vmsk))**2)
		fft_object_F()
		C = np.copy(np.abs(fft_2))

		##Plot ARAR Power
		plt.figure(figsize=(8, 8))
		plt.imshow(
			np.fft.fftshift(C),
			vmin=C_ARAR_min,
			vmax=C_ARAR_max,
			norm=LogNorm(),
			aspect='auto',
			extent=[
				f_d.min().value,
				f_d.max().value,
				tau.max().value,
				tau.min().value
			])
		plt.xlabel(r'$f_D$ (mHz)')
		plt.ylabel(r'$\tau$ (ms)')
		plt.title(r'$C^{%s%s}$' %(t1,t1))
		plt.colorbar()
		plt.plot(f_d, (f_d**2) / B1, 'k', alpha=.5)
		plt.plot(f_d, (f_d**2) / B2, 'k', alpha=.5)
		plt.xlim((-f_arc_max, f_arc_max))
		plt.ylim((0, tau_arc_max))
		plt.savefig('%s/DB_b0834_%s%s_Power_%s.png' % (dirname,t1,t1,dset))
		plt.close('all')

		##Estimate and Remove Noise from Reduced Power Spectrum
		N = C[np.abs(tau) > 5 * tau.max() / 6, :][:, np.abs(f_d) > 5 * f_d.max() / 6].mean()
		fft_2[:]=C - N
		fft_2[fft_2 < N] = 0
		fft_object_B()
		ACOR=np.copy(fft_1)
		Smat=np.zeros((nf2 + nf2//4,nt2 + nt2//4,nf2 + nf2//4,nt2 + nt2//4),dtype=complex)
		Smat[:,:,:,:] = ACOR[
			np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] -
			np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[np.newaxis, np.newaxis, :, np.newaxis],
			np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, :, np.newaxis, np.newaxis] -
			np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, np.newaxis, np.newaxis, :]]
		#Smat=np.reshape(Smat,(nt2*nf2,nt2*nf2))-np.diag(N*np.ones(nt2*nf2))
		Smat=np.reshape(Smat,((nf2 + nf2//4)*(nt2 + nt2//4),(nf2 + nf2//4)*(nt2 + nt2//4)))

	def I_filter(args):
		global ARAR
		global Smat
		global Vmsk
		global N
		global nf2
		global nt2
		global rank
		i,j=args

		if i==0:
			f_start=0
			f_stop=f_start + nf2 + nf2//4
		elif (i+1)*nf2==GBAR.shape[0]:
			f_stop=(i+1)*nf2
			f_start=f_stop - nf2 - nf2//4
		else:
			f_start=i*nf2 - nf2//8
			f_stop=f_start + nf2 + nf2//4
		
		if j==0:
			t_start=0
			t_stop=t_start + nt2 + nt2//4
		elif (j+1)*nt2==GBAR.shape[1]:
			t_stop=(j+1)*nt2
			t_start=t_stop - nt2 - nt2//4
		else:
			t_start=j*nt2 - nt2//8
			t_stop=t_start + nt2 + nt2//4
		ts = MPI.Wtime()
		H = np.tile(Vmsk[t_start:t_stop],nf2+nf2//4)
		L = (Smat * H) @ np.linalg.inv( (Smat * H)*H[:,np.newaxis] + np.diag(N*(100-99*H)))
		filt = np.reshape(
			(L @ ARAR[f_start:f_stop,t_start:t_stop].flatten().T).T,
			(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start].real
		err = np.reshape(
			np.diag(Smat - 2*((Smat * H) @ np.conjugate(L.T))+L @ ((Smat * H)*H[:,np.newaxis] + np.diag(N*(100-99*H)))@ np.conjugate(L.T)),
			(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start]
		print('Rank %s: %s' %(rank,MPI.Wtime()-ts),flush=True)
		return(i,j,filt,err)

	comm.Barrier()
	pool = MPIPool(loadbalance=True)
	if not pool.is_master():
		pool.wait()
	else:
		print('Filter Intensity', flush=True)
		results=pool.map(I_filter,tasks)
		I_filt=np.zeros((nf,nt))
		I_filt_err=np.zeros((nf,nt))
		for i,j,arr,err in results:
			I_filt[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=arr.real
			I_filt_err[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=err.real
		np.save('%s/DB_b0834_mfilt_I_%s.npy' % (dirname,specs),I_filt)
		np.save('%s/DB_b0834_mfilt_I_err_%s.npy' % (dirname,specs),I_filt_err)
	pool.close()
	comm.Barrier()

if rank>0:
	##GBAR Power Spectrum
	fft_1[:] = np.copy(GBAR)
	fft_object_F()
	C = np.abs(fft_2 / np.sqrt(nf*nt))**2
	##Old tests commented out the next 5 lines	
	fft_2[:] = np.copy(C)
	fft_object_B()
	fft_1 /= np.fft.ifft(np.abs(np.fft.fft(Vmsk)/np.sqrt(nt))**2)
	fft_object_F()
	C = np.copy(np.abs(fft_2))

	N = C[np.abs(tau) > 5 * tau.max() / 6, :][:, np.abs(f_d) > 5 * f_d.max() / 6].mean()
	fft_2[:]=C - N
	fft_2[fft_2 < N] = 0
	fft_object_B()
	ACOR=np.copy(fft_1)

	Comp_plotter(GBAR,
		x_start=times[0],
		x_end=times[-1],
		y_start=freqs[0],
		y_end=freqs[-1],
		y_name='Freq',
		x_name='Time',
		title='%s%s Pre-Filter' %(t2,t1),
		xlim=None,
		ylim=None,
		zlim=((-5,5), (-5,5), (None, None)),
		rbinx=1,
		rbiny=1)
	plt.savefig('%s/DB_b0834_%s%s_%s.png' % (dirname,t2,t1,dset))
	plt.close('all')
	
	Smat=np.zeros((nf2 + nf2//4,nt2 + nt2//4,nf2 + nf2//4,nt2 + nt2//4),dtype=complex)
	Smat[:,:,:,:] = ACOR[
		np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] -
		np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[np.newaxis, np.newaxis, :, np.newaxis],
		np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, :, np.newaxis, np.newaxis] -
		np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, np.newaxis, np.newaxis, :]]
	#Smat=np.reshape(Smat,(nt2*nf2,nt2*nf2))-np.diag(N*np.ones(nt2*nf2))
	Smat=np.reshape(Smat,((nf2 + nf2//4)*(nt2 + nt2//4),(nf2 + nf2//4)*(nt2 + nt2//4)))
	if rank==1:
		Comp_plotter(GBAR[nf2:2*nf2,:nt2],
			x_start=times[0],
			x_end=times[nt2],
			y_start=freqs[nf2],
			y_end=freqs[2*nf2],
			y_name='Freq',
			x_name='Time',
			title='%s%s Pre-Filter' %(t2,t1),
			xlim=None,
			ylim=None,
			zlim=((-5,5), (-5,5), (None, None)),
			rbinx=1,
			rbiny=1)
		plt.savefig('%s/DB_b0834_%s%s2_%s.png' % (dirname,t2,t1,dset))
		plt.close('all')
comm.Barrier()
#print(Time.now(),flush=True)
#H = np.tile(Vmsk[:nt2],nf2)
#L = (Smat * H) @ np.linalg.inv( (Smat * H)*H[:,np.newaxis] + np.diag(N*(100-99*H)))
#filt = np.reshape((L @ GBAR[:nf2,:nt2].flatten().T).T,(nf2,nt2))

def V_filter(args):
	global GBAR
	global Smat
	global Vmsk
	global N
	global nf2
	global nt2
	global rank
	i,j=args

	if i==0:
		f_start=0
		f_stop=f_start + nf2 + nf2//4
	elif (i+1)*nf2==GBAR.shape[0]:
		f_stop=(i+1)*nf2
		f_start=f_stop - nf2 - nf2//4
	else:
		f_start=i*nf2 - nf2//8
		f_stop=f_start + nf2 + nf2//4
	
	if j==0:
		t_start=0
		t_stop=t_start + nt2 + nt2//4
	elif (j+1)*nt2==GBAR.shape[1]:
		t_stop=(j+1)*nt2
		t_start=t_stop - nt2 - nt2//4
	else:
		t_start=j*nt2 - nt2//8
		t_stop=t_start + nt2 + nt2//4
	ts = MPI.Wtime()
	H = np.tile(Vmsk[t_start:t_stop],nf2+nf2//4)
	L = (Smat * H) @ np.linalg.inv( (Smat * H)*H[:,np.newaxis] + np.diag(N*(100-99*H)))
	filt = np.reshape(
		(L @ GBAR[f_start:f_stop,t_start:t_stop].flatten().T).T,
		(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start]
	err = np.reshape(
		np.diag(Smat - 2*((Smat * H) @ np.conjugate(L.T))+L @ ((Smat * H)*H[:,np.newaxis] + np.diag(N*(100-99*H)))@ np.conjugate(L.T)),
		(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start]
	print('Rank %s: %s' %(rank,MPI.Wtime()-ts),flush=True)
	return(i,j,filt,err)

if args.s1:
	if rank==0:
		print('Importing Fist Stage Filter')
	V_filt=np.load('%s/DB_b0834_mfilt_V_%s.npy' % (dirname,specs))
	V_filt_err=np.load('%s/DB_b0834_mfilt_V_err_%s.npy' % (dirname,specs))
	N_filt=V_filt_err[:,Vmsk==1].real.mean()
	V_filt*=Vmsk
else:
	pool = MPIPool(loadbalance=True)
	if not pool.is_master():
		pool.wait()
		print('Pool Closed',flush=True)
		N_filt=0.
		V_filt=np.empty((nf,nt),dtype=complex)
	else:
		print('Filtering Visibility (Matrix)',flush=True)
		results=pool.map(V_filter,tasks)
		pool.close()
		V_filt=np.empty((nf,nt),dtype=complex)
		V_filt_err=np.empty((nf,nt),dtype=complex)
		for i,j,arr,err in results:
			V_filt[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=arr
			V_filt_err[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=err
		Comp_plotter(V_filt[nf2:2*nf2,:nt2],
		    x_start=times[0],
		    x_end=times[nt2],
		    y_start=freqs[nf2],
		    y_end=freqs[2*nf2],
		    y_name='Freq',
		    x_name='Time',
		    title='%s%s Matrix Filter' %(t2,t1),
		    xlim=None,
		    ylim=None,
		    zlim=((-5,5), (-5,5), (None, None)),
		    rbinx=1,
		    rbiny=1)
		plt.savefig('%s/DB_b0834_%s%s__mfilt_%s.png' % (dirname,t2,t1,dset))
		plt.close('all')

		Comp_plotter(V_filt_err[nf2:2*nf2,:nt2],
		    x_start=times[0],
		    x_end=times[nt2],
		    y_start=freqs[nf2],
		    y_end=freqs[2*nf2],
		    y_name='Freq',
		    x_name='Time',
		    title='%s%s Matrix Filter Error' %(t2,t1),
		    xlim=None,
		    ylim=None,
		    zlim=((None,None), (None,None), (None, None)),
		    rbinx=1,
		    rbiny=1)
		plt.savefig('%s/DB_b0834_%s%s__mfilt_err_%s.png' % (dirname,t2,t1,dset))
		plt.close('all')

		np.save('%s/DB_b0834_mfilt_V_%s.npy' % (dirname,specs),V_filt)
		np.save('%s/DB_b0834_mfilt_V_err_%s.npy' % (dirname,specs),V_filt_err)
		N_filt=V_filt_err[:,Vmsk==1].real.mean()
		V_filt*=Vmsk
	N_filt=comm.bcast(N_filt,root=0)
	V_filt=comm.bcast(V_filt,root=0)
	print(N_filt,flush=True)
if rank>=1:
	##GBAR Power Spectrum
	fft_1[:] = np.copy(np.angle(V_filt)*Vmsk)
	fft_1[np.abs(V_filt)<np.sqrt(N_filt/2)]=0
	fft_object_F()
	C = np.abs(fft_2 / np.sqrt(nf*nt))**2
	##Old tests commented out the next 5 lines	
	fft_2[:] = np.copy(C)
	fft_object_B()
	fft_1 /= np.fft.ifft(np.abs(np.fft.fft(Vmsk)/np.sqrt(nt))**2)
	fft_object_F()
	C = np.copy(np.abs(fft_2))

	##Estimate and Remove Noise from Reduced Power
	N_phase = C[np.abs(tau) > 5 * tau.max() / 6, :][:, np.abs(f_d) > 5 * f_d.max() / 6].mean()
	print(N,flush=True)
	fft_2[:]=C - N_phase
	fft_2[np.abs(fft_2)<0]=0
	fft_object_B()
	Smat=np.zeros((nf2 + nf2//4,nt2 + nt2//4,nf2 + nf2//4,nt2 + nt2//4),dtype=complex)
	Smat[:,:,:,:] = fft_1[
		np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[:, np.newaxis, np.newaxis, np.newaxis] -
		np.linspace(0, nf2 - 1 + nf2//4, nf2 + nf2//4).astype(int)[np.newaxis, np.newaxis, :, np.newaxis],
		np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, :, np.newaxis, np.newaxis] -
		np.linspace(0, nt2 - 1 + nt2//4, nt2 + nt2//4).astype(int)[np.newaxis, np.newaxis, np.newaxis, :]].real
	#Smat=np.reshape(Smat,(nt2*nf2,nt2*nf2))-np.diag(N*np.ones(nt2*nf2))
	Smat=np.reshape(Smat,((nf2 + nf2//4)*(nt2 + nt2//4),(nf2 + nf2//4)*(nt2 + nt2//4)))
	amps=np.linspace(np.abs(V_filt).min(),np.abs(V_filt).max(),10000)
	var=np.angle((np.random.normal(0,1,(10000,10000))+1j*np.random.normal(0,1,(10000,10000)))*np.sqrt(N_filt/2)+amps).var(0)
	var[amps<np.sqrt(N_filt/2)]=100*np.pi
	var_interp=interp1d(amps,var,fill_value=100*np.pi)
def phase_filter(args):
	global V_filt
	global Smat
	global Vmsk
	global nf2
	global nt2
	global rank
	i,j=args

	if i==0:
		f_start=0
		f_stop=f_start + nf2 + nf2//4
	elif (i+1)*nf2==V_filt.shape[0]:
		f_stop=(i+1)*nf2
		f_start=f_stop - nf2 - nf2//4
	else:
		f_start=i*nf2 - nf2//8
		f_stop=f_start + nf2 + nf2//4
	
	if j==0:
		t_start=0
		t_stop=t_start + nt2 + nt2//4
	elif (j+1)*nt2==V_filt.shape[1]:
		t_stop=(j+1)*nt2
		t_start=t_stop - nt2 - nt2//4
	else:
		t_start=j*nt2 - nt2//8
		t_stop=t_start + nt2 + nt2//4
	ts = MPI.Wtime()
	H = np.tile(Vmsk[t_start:t_stop],nf2 + nf2//4)
	N_arr=np.diag(var_interp(np.abs(V_filt[f_start:f_stop,t_start:t_stop]).flatten()))
	L = (Smat * H) @ np.linalg.inv( (Smat * H)*H[:,np.newaxis] + N_arr)
	filt = np.reshape(
		(L @ (np.angle(V_filt[f_start:f_stop,t_start:t_stop])*Vmsk[t_start:t_stop]).flatten().T).T,
		(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start]
	err = np.reshape(
		np.diag(Smat - 2*((Smat * H) @ np.conjugate(L.T))+L @ ((Smat * H)*H[:,np.newaxis] + N_arr)@ np.conjugate(L.T)),
		(nf2 + nf2//4,nt2 + nt2//4))[i*nf2-f_start:(i+1)*nf2-f_start,j*nt2-t_start:(j+1)*nt2-t_start]
	print('Rank %s: %s' %(rank,MPI.Wtime()-ts),flush=True)
	return(i,j,filt,err)

comm.Barrier()
pool = MPIPool(loadbalance=True)
if not pool.is_master():
	pool.wait()
	sys.exit(0)
else:
	print('Filter Phases', flush=True)
	results=pool.map(phase_filter,tasks)
	phase_filt=np.zeros((nf,nt))
	phase_filt_err=np.zeros((nf,nt))
	for i,j,arr,err in results:
		phase_filt[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=arr.real
		phase_filt_err[i*nf2:(i+1)*nf2,j*nt2:(j+1)*nt2]=err.real
	np.save('%s/DB_b0834_phasefilt_%s.npy' % (dirname,specs),phase_filt)
	fig, axes = plt.subplots(
		nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8, 8))
	im00 = axes[0].imshow(
		np.angle(V_filt[nf2:2*nf2,:nt2]),
		aspect='auto',
		extent= [times[0].value, times[nt2].value, freqs[2*nf2].value, freqs[nf2].value],
		vmin=-np.pi,
		vmax=np.pi,
		cmap='twilight')
	axes[0].set_title('Phase After 1st Filter')
	im10 = axes[1].imshow(
		phase_filt[nf2:2*nf2,:nt2],
		aspect='auto',
		extent= [times[0].value, times[nt2].value, freqs[2*nf2].value, freqs[nf2].value],
		vmin=-np.pi,
		vmax=np.pi,
		cmap='twilight')
	axes[1].set_title('Phase After 2nd Filter')
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	plt.colorbar(im00, cax=cbar_ax)
	axes[1].set_xlabel('Time (s)')
	axes[0].set_ylabel('Freq (MHz)')
	plt.savefig('%s/DB_b0834_%s%s_phasefilt_%s.png' % (dirname,t2,t1,dset))
	plt.close('all')
pool.close()

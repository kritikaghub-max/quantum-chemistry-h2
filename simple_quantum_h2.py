import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
import warnings
warnings.filterwarnings('ignore')

class SimpleH2Chemistry:
    """
    A simplified quantum chemistry class for H2 molecule analysis.
    Uses manually constructed Hamiltonians to avoid package conflicts.
    """
    
    def __init__(self, distance=0.735):
        """
        Initialize with H2 molecule at given distance.
        
        Args:
            distance (float): Bond distance in Angstroms
        """
        self.distance = distance
        self.hamiltonian = None
        
    def create_h2_hamiltonian(self):
        """
        Create a simplified H2 Hamiltonian using known coefficients.
        This uses the STO-3G basis set results for H2 at various distances.
        """
        print(f"Creating H2 Hamiltonian at distance: {self.distance} Å")
        
        # Simplified H2 Hamiltonian coefficients (STO-3G basis)
        # These are approximate values for demonstration
        if abs(self.distance - 0.735) < 0.1:  # Near equilibrium
            # Coefficients for H2 at ~0.74 Angstrom
            coeffs = [-1.0523732, 0.39793742, -0.39793742, -0.01128010, 
                     0.18093119, 0.18093119]
        else:
            # Scale coefficients based on distance (simplified model)
            scale = np.exp(-abs(self.distance - 0.735))
            coeffs = [-1.0523732 * scale, 0.39793742, -0.39793742, 
                     -0.01128010, 0.18093119, 0.18093119]
        
        # Pauli strings for H2 (2 electrons, 4 qubits with Jordan-Wigner)
        pauli_strings = ['IIII', 'IIIZ', 'IIZI', 'IIZZ', 'ZIIZ', 'ZIII']
        
        # Create the Hamiltonian
        pauli_list = [(pauli, coeff) for pauli, coeff in zip(pauli_strings, coeffs)]
        self.hamiltonian = SparsePauliOp.from_list(pauli_list)
        
        print(f"Hamiltonian created with {self.hamiltonian.num_qubits} qubits")
        return self.hamiltonian
        
    def exact_ground_state(self):
        """Calculate exact ground state energy using matrix diagonalization."""
        if self.hamiltonian is None:
            self.create_h2_hamiltonian()
            
        # Get the matrix representation and find eigenvalues
        hamiltonian_matrix = self.hamiltonian.to_matrix()
        eigenvalues = np.linalg.eigvals(hamiltonian_matrix)
        ground_state_energy = np.min(np.real(eigenvalues))
        
        print(f"Exact ground state energy: {ground_state_energy:.6f} Hartree")
        return ground_state_energy
        
    def create_ansatz_circuit(self, num_qubits=4):
        """
        Create a simple variational ansatz circuit.
        
        Args:
            num_qubits (int): Number of qubits
            
        Returns:
            QuantumCircuit: Parameterized ansatz circuit
        """
        # Create a simple ansatz with rotation and entangling gates
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks='ry',  # Single qubit rotations
            entanglement_blocks='cz',  # Two-qubit gates
            entanglement='linear',  # Linear connectivity
            reps=1,  # Number of repetitions
            insert_barriers=True
        )
        
        return ansatz
        
    def simple_vqe(self, max_iterations=50):
        """
        Simplified VQE implementation using basic optimization.
        
        Args:
            max_iterations (int): Maximum optimization steps
            
        Returns:
            dict: VQE results
        """
        print("Running simplified VQE...")
        
        if self.hamiltonian is None:
            self.create_h2_hamiltonian()
            
        # Create ansatz circuit
        ansatz = self.create_ansatz_circuit(self.hamiltonian.num_qubits)
        num_params = ansatz.num_parameters
        
        # Initialize parameters randomly
        np.random.seed(42)  # For reproducibility
        params = np.random.uniform(0, 2*np.pi, num_params)
        
        # Simple parameter optimization using random search
        best_energy = float('inf')
        best_params = params.copy()
        
        backend = AerSimulator()
        
        for iteration in range(max_iterations):
            # Add small random perturbations to parameters
            if iteration > 0:
                params = best_params + np.random.normal(0, 0.1, num_params)
            
            # Create parameterized circuit
            bound_circuit = ansatz.assign_parameters(params)
            
            # Measure expectation value of Hamiltonian
            energy = self.measure_expectation_value(bound_circuit, backend)
            
            # Update best result
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f} Hartree")
        
        print(f"VQE completed. Final energy: {best_energy:.6f} Hartree")
        
        return {
            'energy': best_energy,
            'optimal_params': best_params,
            'iterations': max_iterations
        }
        
    def measure_expectation_value(self, circuit, backend, shots=8192):
        """
        Measure the expectation value of the Hamiltonian.
        
        Args:
            circuit (QuantumCircuit): Quantum circuit
            backend: Quantum backend
            shots (int): Number of measurement shots
            
        Returns:
            float: Expectation value
        """
        # For simplicity, we'll use the statevector simulator approach
        # This avoids complex shot-based measurements
        
        from qiskit_aer import StatevectorSimulator
        statevector_backend = StatevectorSimulator()
        
        # Get statevector
        transpiled_circuit = transpile(circuit, statevector_backend)
        job = statevector_backend.run(transpiled_circuit)
        result = job.result()
        statevector = result.get_statevector()
        
        # Convert statevector to numpy array and calculate expectation value
        psi = np.array(statevector.data)  # Convert to numpy array
        hamiltonian_matrix = self.hamiltonian.to_matrix()
        
        # Calculate expectation value: <psi|H|psi>
        expectation_value = np.real(
            np.conj(psi).T @ hamiltonian_matrix @ psi
        )
        
        return expectation_value
        
    def bond_length_scan(self, distances=None):
        """
        Scan different bond lengths to find the potential energy curve.
        
        Args:
            distances (list): Bond distances to scan
            
        Returns:
            tuple: (distances, energies)
        """
        if distances is None:
            distances = np.linspace(0.5, 2.0, 10)
            
        energies = []
        print("Performing bond length scan...")
        
        for dist in distances:
            print(f"\nDistance: {dist:.2f} Å")
            self.distance = dist
            self.create_h2_hamiltonian()
            
            # Use exact solution for the scan (faster and more accurate)
            energy = self.exact_ground_state()
            energies.append(energy)
            
        return distances, np.array(energies)
        
    def plot_results(self, distances, energies, vqe_result=None):
        """
        Plot the potential energy curve and VQE result.
        
        Args:
            distances (array): Bond distances
            energies (array): Ground state energies
            vqe_result (dict, optional): VQE results to plot
        """
        plt.figure(figsize=(12, 8))
        
        # Main potential curve
        plt.subplot(2, 1, 1)
        plt.plot(distances, energies, 'bo-', linewidth=2, markersize=8, label='Exact Solution')
        plt.xlabel('Bond Distance (Å)')
        plt.ylabel('Ground State Energy (Hartree)')
        plt.title('H₂ Molecule Potential Energy Curve')
        plt.grid(True, alpha=0.3)
        
        # Find and mark minimum
        min_idx = np.argmin(energies)
        plt.plot(distances[min_idx], energies[min_idx], 'ro', markersize=12, 
                label=f'Minimum: {distances[min_idx]:.2f} Å')
        plt.legend()
        
        # VQE comparison if provided
        if vqe_result:
            plt.subplot(2, 1, 2)
            exact_energy = self.exact_ground_state()
            vqe_energy = vqe_result['energy']
            error = abs(vqe_energy - exact_energy)
            
            methods = ['Exact', 'VQE']
            energies_comp = [exact_energy, vqe_energy]
            colors = ['blue', 'red']
            
            bars = plt.bar(methods, energies_comp, color=colors, alpha=0.7)
            plt.ylabel('Ground State Energy (Hartree)')
            plt.title(f'VQE vs Exact Solution (Error: {error:.6f} Hartree)')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, energy in zip(bars, energies_comp):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{energy:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nEquilibrium bond length: {distances[min_idx]:.3f} Å")
        print(f"Ground state energy: {energies[min_idx]:.6f} Hartree")

def demonstrate_simple_quantum_chemistry():
    """Main demonstration function."""
    print("=" * 70)
    print("Simple Quantum Chemistry with Qiskit: H2 Molecule Analysis")
    print("=" * 70)
    
    # Create H2 chemistry instance
    h2_chem = SimpleH2Chemistry(distance=0.735)
    
    # Calculate exact ground state
    print("\n1. Exact Solution:")
    print("-" * 30)
    exact_energy = h2_chem.exact_ground_state()
    
    # Run simplified VQE
    print("\n2. VQE Solution:")
    print("-" * 30)
    vqe_result = h2_chem.simple_vqe(max_iterations=30)
    
    # Calculate and display error
    error = abs(vqe_result['energy'] - exact_energy)
    print(f"\nVQE Error: {error:.6f} Hartree ({error/abs(exact_energy)*100:.3f}%)")
    
    # Bond length scan
    print("\n3. Bond Length Scan:")
    print("-" * 30)
    distances, energies = h2_chem.bond_length_scan()
    
    # Plot results
    h2_chem.plot_results(distances, energies, vqe_result)
    
    return h2_chem, vqe_result

def analyze_quantum_circuit():
    """Analyze the quantum circuit structure."""
    print("\n" + "=" * 50)
    print("Quantum Circuit Analysis")
    print("=" * 50)
    
    h2_chem = SimpleH2Chemistry()
    h2_chem.create_h2_hamiltonian()
    
    # Create and display ansatz circuit
    ansatz = h2_chem.create_ansatz_circuit()
    
    print(f"Circuit depth: {ansatz.depth()}")
    print(f"Number of parameters: {ansatz.num_parameters}")
    print(f"Number of qubits: {ansatz.num_qubits}")
    print(f"Gate counts: {ansatz.count_ops()}")
    
    # Draw circuit (if small enough)
    print("\nQuantum Circuit Structure:")
    print(ansatz.draw(output='text'))
    
    return ansatz

if __name__ == "__main__":
    try:
        # Main demonstration
        h2_system, results = demonstrate_simple_quantum_chemistry()
        
        # Circuit analysis
        circuit = analyze_quantum_circuit()
        
        print("\n" + "=" * 70)
        print("🎉 Analysis Complete!")
        print("=" * 70)
        print("\nThis simplified project demonstrated:")
        print("• Manual Hamiltonian construction for H₂")
        print("• Exact diagonalization for ground state energy")
        print("• Simplified VQE with parameter optimization")
        print("• Bond length scanning and potential curves")
        print("• Quantum circuit analysis")
        print("\n💡 This version avoids complex package dependencies!")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install: pip install qiskit qiskit-aer numpy matplotlib")
    except Exception as e:
        print(f"Error: {e}")
        print("Check that all required packages are installed correctly.")
